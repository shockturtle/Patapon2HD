#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import typer
from rich import print
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    # SpinnerColumn,
    TaskID,
    TimeRemainingColumn,
)

import utils.architecture as arch
import utils.dataops as ops


class SeamlessOptions(str, Enum):
    tile = "tile"
    mirror = "mirror"
    replicate = "replicate"
    alpha_pad = "alpha_pad"


class AlphaOptions(str, Enum):
    no_alpha = "no_alpha"
    bas = "bas"
    alpha_separately = "alpha_separately"
    swapping = "swapping"


class Upscale:
    model_str: str = None
    input: Path = None
    output: Path = None
    reverse: bool = None
    skip_existing: bool = None
    delete_input: bool = None
    seamless: SeamlessOptions = None
    cpu: bool = None
    fp16: bool = None
    # device_id: int = None
    cache_max_split_depth: bool = None
    binary_alpha: bool = None
    ternary_alpha: bool = None
    alpha_threshold: float = None
    alpha_boundary_offset: float = None
    alpha_mode: AlphaOptions = None
    log: logging.Logger = None

    device: torch.device = None
    in_nc: int = None
    out_nc: int = None
    last_model: str = None
    last_in_nc: int = None
    last_out_nc: int = None
    last_nf: int = None
    last_nb: int = None
    last_scale: int = None
    last_kind: str = None
    model: Union[arch.nn.Module, arch.RRDBNet, arch.SPSRNet] = None

    def __init__(
        self,
        model: str,
        input: Path,
        output: Path,
        reverse: bool = False,
        skip_existing: bool = False,
        delete_input: bool = False,
        seamless: Optional[SeamlessOptions] = None,
        cpu: bool = False,
        fp16: bool = False,
        device_id: int = 0,
        cache_max_split_depth: bool = False,
        binary_alpha: bool = False,
        ternary_alpha: bool = False,
        alpha_threshold: float = 0.5,
        alpha_boundary_offset: float = 0.2,
        alpha_mode: Optional[AlphaOptions] = None,
        log: logging.Logger = logging.getLogger(),
    ) -> None:
        self.model_str = model
        self.input = input.resolve()
        self.output = output.resolve()
        self.reverse = reverse
        self.skip_existing = skip_existing
        self.delete_input = delete_input
        self.seamless = seamless
        self.cpu = cpu
        self.fp16 = fp16
        self.device = torch.device("cpu" if self.cpu else f"cuda:{device_id}")
        self.cache_max_split_depth = cache_max_split_depth
        self.binary_alpha = binary_alpha
        self.ternary_alpha = ternary_alpha
        self.alpha_threshold = alpha_threshold
        self.alpha_boundary_offset = alpha_boundary_offset
        self.alpha_mode = alpha_mode
        self.log = log
        if self.fp16:
            torch.set_default_tensor_type(
                torch.HalfTensor if self.cpu else torch.cuda.HalfTensor
            )

    def run(self) -> None:
        model_chain = (
            self.model_str.split("+")
            if "+" in self.model_str
            else self.model_str.split(">")
        )

        for idx, model in enumerate(model_chain):

            interpolations = (
                model.split("|") if "|" in self.model_str else model.split("&")
            )

            if len(interpolations) > 1:
                for i, interpolation in enumerate(interpolations):
                    interp_model, interp_amount = (
                        interpolation.split("@")
                        if "@" in interpolation
                        else interpolation.split(":")
                    )
                    interp_model = self.__check_model_path(interp_model)
                    interpolations[i] = f"{interp_model}@{interp_amount}"
                model_chain[idx] = "&".join(interpolations)
            else:
                model_chain[idx] = self.__check_model_path(model)

        if not self.input.exists():
            self.log.error(f'Folder "{self.input}" does not exist.')
            sys.exit(1)
        elif self.input.is_file():
            self.log.error(f'Folder "{self.input}" is a file.')
            sys.exit(1)
        elif self.output.is_file():
            self.log.error(f'Folder "{self.output}" is a file.')
            sys.exit(1)
        elif not self.output.exists():
            self.output.mkdir(parents=True)

        self.in_nc = None
        self.out_nc = None

        print(
            'Model{:s}: "{:s}"'.format(
                "s" if len(model_chain) > 1 else "",
                # ", ".join([Path(x).stem for x in model_chain]),
                ", ".join([x for x in model_chain]),
            )
        )

        images: List[Path] = []
        for ext in ["png", "jpg", "jpeg", "gif", "bmp", "tiff", "tga"]:
            images.extend(self.input.glob(f"**/*.{ext}"))

        # Store the maximum split depths for each model in the chain
        # TODO: there might be a better way of doing this but it's good enough for now
        split_depths = {}

        with Progress(
            # SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
        ) as progress:
            task_upscaling = progress.add_task("Upscaling", total=len(images))
            for idx, img_path in enumerate(images, 1):
                img_input_path_rel = img_path.relative_to(self.input)
                output_dir = self.output.joinpath(img_input_path_rel).parent
                img_output_path_rel = output_dir.joinpath(f"{img_path.stem}.png")
                output_dir.mkdir(parents=True, exist_ok=True)
                if len(model_chain) == 1:
                    self.log.info(
                        f'Processing {str(idx).zfill(len(str(len(images))))}: "{img_input_path_rel}"'
                    )
                if self.skip_existing and img_output_path_rel.is_file():
                    self.log.warning("Already exists, skipping")
                    if self.delete_input:
                        img_path.unlink(missing_ok=True)
                    progress.advance(task_upscaling)
                    continue
                # read image
                img = cv2.imread(str(img_path.absolute()), cv2.IMREAD_UNCHANGED)
                if len(img.shape) < 3:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # Seamless modes
                if self.seamless == SeamlessOptions.tile:
                    img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_WRAP)
                elif self.seamless == SeamlessOptions.mirror:
                    img = cv2.copyMakeBorder(
                        img, 16, 16, 16, 16, cv2.BORDER_REFLECT_101
                    )
                elif self.seamless == SeamlessOptions.replicate:
                    img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REPLICATE)
                elif self.seamless == SeamlessOptions.alpha_pad:
                    img = cv2.copyMakeBorder(
                        img, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0]
                    )
                final_scale: int = 1

                task_model_chain: TaskID = None
                if len(model_chain) > 1:
                    task_model_chain = progress.add_task(
                        f'{str(idx).zfill(len(str(len(images))))} - "{img_input_path_rel}"',
                        total=len(model_chain),
                    )
                for i, model_path in enumerate(model_chain):

                    img_height, img_width = img.shape[:2]

                    # Load the model so we can access the scale
                    self.load_model(model_path)

                    if self.cache_max_split_depth and len(split_depths.keys()) > 0:
                        rlt, depth = ops.auto_split_upscale(
                            img,
                            self.upscale,
                            self.last_scale,
                            max_depth=split_depths[i],
                        )
                    else:
                        rlt, depth = ops.auto_split_upscale(
                            img, self.upscale, self.last_scale
                        )
                        split_depths[i] = depth

                    final_scale *= self.last_scale

                    # This is for model chaining
                    img = rlt.astype("uint8")
                    if len(model_chain) > 1:
                        progress.advance(task_model_chain)

                if self.seamless:
                    rlt = self.crop_seamless(rlt, final_scale)

                cv2.imwrite(str(img_output_path_rel.absolute()), rlt)

                if self.delete_input:
                    img_path.unlink(missing_ok=True)

                progress.advance(task_upscaling)

    def __check_model_path(self, model_path: str) -> str:
        if Path(model_path).is_file():
            return model_path
        elif Path("./models/").joinpath(model_path).is_file():
            return str(Path("./models/").joinpath(model_path))
        else:
            self.log.error(f'Model "{model_path}" does not exist.')
            sys.exit(1)

    # This code is a somewhat modified version of BlueAmulet's fork of ESRGAN by Xinntao
    def process(self, img: np.ndarray):
        """
        Does the processing part of ESRGAN. This method only exists because the same block of code needs to be ran twice for images with transparency.

                Parameters:
                        img (array): The image to process

                Returns:
                        rlt (array): The processed image
        """
        if img.shape[2] == 3:
            img = img[:, :, [2, 1, 0]]
        elif img.shape[2] == 4:
            img = img[:, :, [2, 1, 0, 3]]
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        if self.fp16:
            img = img.half()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(self.device)

        output = self.model(img_LR).data.squeeze(0).float().cpu().clamp_(0, 1).numpy()
        if output.shape[0] == 3:
            output = output[[2, 1, 0], :, :]
        elif output.shape[0] == 4:
            output = output[[2, 1, 0, 3], :, :]
        output = np.transpose(output, (1, 2, 0))
        return output

    def load_model(self, model_path: str):
        if model_path != self.last_model:
            # interpolating OTF, example: 4xBox:25&4xPSNR:75
            if (":" in model_path or "@" in model_path) and (
                "&" in model_path or "|" in model_path
            ):
                interps = model_path.split("&")[:2]
                model_1 = torch.load(interps[0].split("@")[0])
                model_2 = torch.load(interps[1].split("@")[0])
                state_dict = OrderedDict()
                f = open("demofile2.txt", "a")
                f.write(repr(model_2))
                f.close()
                for k, v_1 in model_1.items():
                    v_2 = model_2[k]
                    state_dict[k] = (int(interps[0].split("@")[1]) / 100) * v_1 + (
                        int(interps[1].split("@")[1]) / 100
                    ) * v_2
            else:
                state_dict = torch.load(model_path)

            if "conv_first.weight" in state_dict:
                print("Attempting to convert and load a new-format model")
                old_net = {}
                items = []
                for k, v in state_dict.items():
                    items.append(k)

                old_net["model.0.weight"] = state_dict["conv_first.weight"]
                old_net["model.0.bias"] = state_dict["conv_first.bias"]

                for k in items.copy():
                    if "RDB" in k:
                        ori_k = k.replace("RRDB_trunk.", "model.1.sub.")
                        if ".weight" in k:
                            ori_k = ori_k.replace(".weight", ".0.weight")
                        elif ".bias" in k:
                            ori_k = ori_k.replace(".bias", ".0.bias")
                        old_net[ori_k] = state_dict[k]
                        items.remove(k)

                old_net["model.1.sub.23.weight"] = state_dict["trunk_conv.weight"]
                old_net["model.1.sub.23.bias"] = state_dict["trunk_conv.bias"]
                old_net["model.3.weight"] = state_dict["upconv1.weight"]
                old_net["model.3.bias"] = state_dict["upconv1.bias"]
                old_net["model.6.weight"] = state_dict["upconv2.weight"]
                old_net["model.6.bias"] = state_dict["upconv2.bias"]
                old_net["model.8.weight"] = state_dict["HRconv.weight"]
                old_net["model.8.bias"] = state_dict["HRconv.bias"]
                old_net["model.10.weight"] = state_dict["conv_last.weight"]
                old_net["model.10.bias"] = state_dict["conv_last.bias"]
                state_dict = old_net

            # extract model information
            scale2 = 0
            max_part = 0
            plus = False
            if "f_HR_conv1.0.weight" in state_dict:
                kind = "SPSR"
                scalemin = 4
            else:
                kind = "ESRGAN"
                scalemin = 6
            for part in list(state_dict):
                parts = part.split(".")
                n_parts = len(parts)
                if n_parts == 5 and parts[2] == "sub":
                    nb = int(parts[3])
                elif n_parts == 3:
                    part_num = int(parts[1])
                    if (
                        part_num > scalemin
                        and parts[0] == "model"
                        and parts[2] == "weight"
                    ):
                        scale2 += 1
                    if part_num > max_part:
                        max_part = part_num
                        self.out_nc = state_dict[part].shape[0]
                if "conv1x1" in part and not plus:
                    plus = True

            upscale = 2 ** scale2
            self.in_nc = state_dict["model.0.weight"].shape[1]
            if kind == "SPSR":
                self.out_nc = state_dict["f_HR_conv1.0.weight"].shape[0]
            nf = state_dict["model.0.weight"].shape[0]

            if (
                self.in_nc != self.last_in_nc
                or self.out_nc != self.last_out_nc
                or nf != self.last_nf
                or nb != self.last_nb
                or upscale != self.last_scale
                or kind != self.last_kind
            ):
                if kind == "ESRGAN":
                    self.model = arch.RRDBNet(
                        in_nc=self.in_nc,
                        out_nc=self.out_nc,
                        nf=nf,
                        nb=nb,
                        gc=32,
                        upscale=upscale,
                        norm_type=None,
                        act_type="leakyrelu",
                        mode="CNA",
                        upsample_mode="upconv",
                        plus=plus,
                    )
                elif kind == "SPSR":
                    self.model = arch.SPSRNet(
                        self.in_nc,
                        self.out_nc,
                        nf,
                        nb,
                        gc=32,
                        upscale=upscale,
                        norm_type=None,
                        act_type="leakyrelu",
                        mode="CNA",
                        upsample_mode="upconv",
                    )
                self.last_in_nc = self.in_nc
                self.last_out_nc = self.out_nc
                self.last_nf = nf
                self.last_nb = nb
                self.last_scale = upscale
                self.last_kind = kind
                self.last_model = model_path

            self.model.load_state_dict(state_dict, strict=True)
            del state_dict
            self.model.eval()
            for k, v in self.model.named_parameters():
                v.requires_grad = False
            self.model = self.model.to(self.device)
        self.last_model = model_path

    # This code is a somewhat modified version of BlueAmulet's fork of ESRGAN by Xinntao
    def upscale(self, img: np.ndarray) -> np.ndarray:
        """
        Upscales the image passed in with the specified model

                Parameters:
                        img: The image to upscale
                        model_path (string): The model to use

                Returns:
                        output: The processed image
        """

        img = img * 1.0 / np.iinfo(img.dtype).max

        if (
            img.ndim == 3
            and img.shape[2] == 4
            and self.last_in_nc == 3
            and self.last_out_nc == 3
        ):

            # Fill alpha with white and with black, remove the difference
            if self.alpha_mode == AlphaOptions.bas:
                img1 = np.copy(img[:, :, :3])
                img2 = np.copy(img[:, :, :3])
                for c in range(3):
                    img1[:, :, c] *= img[:, :, 3]
                    img2[:, :, c] = (img2[:, :, c] - 1) * img[:, :, 3] + 1

                output1 = self.process(img1)
                output2 = self.process(img2)
                alpha = 1 - np.mean(output2 - output1, axis=2)
                output = np.dstack((output1, alpha))
                output = np.clip(output, 0, 1)
            # Upscale the alpha channel itself as its own image
            elif self.alpha_mode == AlphaOptions.alpha_separately:
                img1 = np.copy(img[:, :, :3])
                img2 = cv2.merge((img[:, :, 3], img[:, :, 3], img[:, :, 3]))
                output1 = self.process(img1)
                output2 = self.process(img2)
                output = cv2.merge(
                    (
                        output1[:, :, 0],
                        output1[:, :, 1],
                        output1[:, :, 2],
                        output2[:, :, 0],
                    )
                )
            # Use the alpha channel like a regular channel
            elif self.alpha_mode == AlphaOptions.swapping:
                img1 = cv2.merge((img[:, :, 0], img[:, :, 1], img[:, :, 2]))
                img2 = cv2.merge((img[:, :, 1], img[:, :, 2], img[:, :, 3]))
                output1 = self.process(img1)
                output2 = self.process(img2)
                output = cv2.merge(
                    (
                        output1[:, :, 0],
                        output1[:, :, 1],
                        output1[:, :, 2],
                        output2[:, :, 2],
                    )
                )
            # Remove alpha
            else:
                img1 = np.copy(img[:, :, :3])
                output = self.process(img1)
                output = cv2.cvtColor(output, cv2.COLOR_BGR2BGRA)

            if self.binary_alpha:
                alpha = output[:, :, 3]
                threshold = self.alpha_threshold
                _, alpha = cv2.threshold(alpha, threshold, 1, cv2.THRESH_BINARY)
                output[:, :, 3] = alpha
            elif self.ternary_alpha:
                alpha = output[:, :, 3]
                half_transparent_lower_bound = (
                    self.alpha_threshold - self.alpha_boundary_offset
                )
                half_transparent_upper_bound = (
                    self.alpha_threshold + self.alpha_boundary_offset
                )
                alpha = np.where(
                    alpha < half_transparent_lower_bound,
                    0,
                    np.where(alpha <= half_transparent_upper_bound, 0.5, 1),
                )
                output[:, :, 3] = alpha
        else:
            if img.ndim == 2:
                img = np.tile(
                    np.expand_dims(img, axis=2), (1, 1, min(self.last_in_nc, 3))
                )
            if img.shape[2] > self.last_in_nc:  # remove extra channels
                self.log.warning("Truncating image channels")
                img = img[:, :, : self.last_in_nc]
            # pad with solid alpha channel
            elif img.shape[2] == 3 and self.last_in_nc == 4:
                img = np.dstack((img, np.full(img.shape[:-1], 1.0)))
            output = self.process(img)

        output = (output * 255.0).round()

        return output

    def crop_seamless(self, img: np.ndarray, scale: int) -> np.ndarray:
        img_height, img_width = img.shape[:2]
        y, x = 16 * scale, 16 * scale
        h, w = img_height - (32 * scale), img_width - (32 * scale)
        img = img[y : y + h, x : x + w]
        return img


app = typer.Typer()


@app.command()
def main(
    model: str = typer.Argument(...),
    input: Path = typer.Option(Path("input"), "--input", "-i", help="Input folder"),
    output: Path = typer.Option(Path("output"), "--output", "-o", help="Output folder"),
    reverse: bool = typer.Option(False, "--reverse", "-r", help="Reverse Order"),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        "-se",
        help="Skip existing output files",
    ),
    delete_input: bool = typer.Option(
        False,
        "--delete-input",
        "-di",
        help="Delete input files after upscaling",
    ),
    seamless: SeamlessOptions = typer.Option(
        None,
        "--seamless",
        "-s",
        case_sensitive=False,
        help="Helps seamlessly upscale an image. tile = repeating along edges. mirror = reflected along edges. replicate = extended pixels along edges. alpha_pad = extended alpha border.",
    ),
    cpu: bool = typer.Option(False, "--cpu", "-c", help="Use CPU instead of CUDA"),
    fp16: bool = typer.Option(
        False,
        "--floating-point-16",
        "-fp16",
        help="Use FloatingPoint16/Halftensor type for images.",
    ),
    device_id: int = typer.Option(
        0, "--device-id", "-did", help="The numerical ID of the GPU you want to use."
    ),
    cache_max_split_depth: bool = typer.Option(
        False,
        "--cache-max-split-depth",
        "-cmsd",
        help="Caches the maximum recursion depth used by the split/merge function. Useful only when upscaling images of the same size.",
    ),
    binary_alpha: bool = typer.Option(
        False,
        "--binary-alpha",
        "-ba",
        help="Whether to use a 1 bit alpha transparency channel, Useful for PSX upscaling",
    ),
    ternary_alpha: bool = typer.Option(
        False,
        "--ternary-alpha",
        "-ta",
        help="Whether to use a 2 bit alpha transparency channel, Useful for PSX upscaling",
    ),
    alpha_threshold: float = typer.Option(
        0.5,
        "--alpha-threshold",
        "-at",
        help="Only used when binary_alpha is supplied. Defines the alpha threshold for binary transparency",
    ),
    alpha_boundary_offset: float = typer.Option(
        0.2,
        "--alpha-boundary-offset",
        "-abo",
        help="Only used when binary_alpha is supplied. Determines the offset boundary from the alpha threshold for half transparency.",
    ),
    alpha_mode: AlphaOptions = typer.Option(
        None,
        "--alpha-mode",
        "-am",
        help="Type of alpha processing to use. no_alpha = is no alpha processing. bas = is BA's difference method. alpha_separately = is upscaling the alpha channel separately (like IEU). swapping = is swapping an existing channel with the alpha channel.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose mode",
    ),
):

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True)],
        # handlers=[RichHandler(markup=True, rich_tracebacks=True)],
    )

    upscale = Upscale(
        model=model,
        input=input,
        output=output,
        reverse=reverse,
        skip_existing=skip_existing,
        delete_input=delete_input,
        seamless=seamless,
        cpu=cpu,
        fp16=fp16,
        device_id=device_id,
        cache_max_split_depth=cache_max_split_depth,
        binary_alpha=binary_alpha,
        ternary_alpha=ternary_alpha,
        alpha_threshold=alpha_threshold,
        alpha_boundary_offset=alpha_boundary_offset,
        alpha_mode=alpha_mode,
    )
    upscale.run()


if __name__ == "__main__":
    app()
