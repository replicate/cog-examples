import tempfile
import os
from pathlib import Path
import shutil

import cog
import skvideo.io
import moviepy.editor
from queue import Queue
import numpy as np
import _thread
import cv2
import torch
from torch.nn import functional as F

from benchmark.pytorch_msssim import ssim_matlab
from model.RIFE_HDv2 import Model as RIFEModel


class Model(cog.Model):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        self.model = RIFEModel()
        self.model.load_model("train_log", -1)
        self.model.device()
        self.model.eval()

    @cog.input(
        "img1",
        type=Path,
        default=None,
        help="First image when interpolating between two images (requires both img1 and img2 to be set)",
    )
    @cog.input(
        "img2",
        type=Path,
        default=None,
        help="Second image when interpolating between two images (requires both img1 and img2 to be set)",
    )
    @cog.input(
        "ratio",
        type=float,
        default=0,
        help="Inference ratio between two images with 0 - 1 range",
    )
    @cog.input(
        "rthreshold",
        type=float,
        default=0.02,
        help="returns image when actual ratio falls in given range threshold",
    )
    @cog.input(
        "rmaxcycles", type=int, default=8, help="limit max number of bisectional cycles"
    )
    @cog.input("video", type=Path, default=None, help="Input video")
    @cog.input("montage", type=Path, default=None, help="Montage origin video")
    @cog.input("uhd", type=bool, default=False, help="Support 4k video")
    @cog.input(
        "skip",
        type=bool,
        default=False,
        help="Whether to remove static frames before processing",
    )
    @cog.input('scale', default=1.0, type=float, help='Try scale=0.5 for 4k video')
    @cog.input("fps", type=int, default=None, help="Output frame rate")
    @cog.input(
        "png", type=bool, default=False, help="Whether to vid_out png format vid_outs"
    )
    @cog.input("ext", type=str, default="mp4", help="Output video extension")
    @cog.input("exp", type=int, default=1, help="Slowdown factor (e.g. 1, 2, 4)")
    def run(
            self, img1, img2, ratio, rthreshold, rmaxcycles, video, montage, uhd, scale, skip, fps, png, ext, exp
    ):
        if (img1 is not None and img2 is None) or (img1 is None and img2 is not None):
            raise ValueError("img1 and img2 must both be set if one of them is set")
        if img1 is not None and (
            video is not None or montage is not None
        ):
            raise ValueError(
                "Either img1/img2 or video/montage must be set, not both"
            )
        if img1 is not None:
            return self.run_images(img1, img2, ratio, rthreshold, rmaxcycles, exp)
        if video is not None:
            return self.run_video(video, montage, uhd, scale, skip, fps, png, ext, exp)
        raise ValueError("Either img1/img2 or video/montage must be set")

    def run_images(self, img1, img2, ratio, rthreshold, rmaxcycles, exp):
        img1, img2 = str(img1), str(img2)
        if img1.endswith(".exr") and img2.endswith(".exr"):
            is_exr = True
            img1 = cv2.imread(img1, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            img2 = cv2.imread(img2, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(self.device)).unsqueeze(0)
            img2 = (torch.tensor(img2.transpose(2, 0, 1)).to(self.device)).unsqueeze(0)
        else:
            is_exr = False
            img1 = cv2.imread(img1)
            img2 = cv2.imread(img2)
            img1 = (
                torch.tensor(img1.transpose(2, 0, 1)).to(self.device) / 255.0
            ).unsqueeze(0)
            img2 = (
                torch.tensor(img2.transpose(2, 0, 1)).to(self.device) / 255.0
            ).unsqueeze(0)

        n, c, h, w = img1.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img1 = F.pad(img1, padding)
        img2 = F.pad(img2, padding)

        if ratio > 0:
            img_list = [img1]
            img1_ratio = 0.0
            img2_ratio = 1.0
            if ratio <= img1_ratio + rthreshold / 2:
                middle = img1
            elif ratio >= img2_ratio - rthreshold / 2:
                middle = img2
            else:
                tmp_img1 = img1
                tmp_img2 = img2
                for inference_cycle in range(rmaxcycles):
                    middle = self.model.inference(tmp_img1, tmp_img2)
                    middle_ratio = ( img1_ratio + img2_ratio ) / 2
                    if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
                        break
                    if ratio > middle_ratio:
                        tmp_img1 = middle
                        img1_ratio = middle_ratio
                    else:
                        tmp_img2 = middle
                        img2_ratio = middle_ratio
            img_list.append(middle)
            img_list.append(img2)
        else:
            img_list = [img1, img2]
            for i in range(exp):
                tmp = []
                for j in range(len(img_list) - 1):
                    mid = self.model.inference(img_list[j], img_list[j + 1])
                    tmp.append(img_list[j])
                    tmp.append(mid)
                tmp.append(img2)
                img_list = tmp

        tmpdir = tempfile.mkdtemp()
        output_dir = os.path.join(tmpdir, "output")
        os.mkdir(output_dir)

        for i in range(len(img_list)):
            if is_exr:
                cv2.imwrite(f'{output_dir}/img{i}.exr', (img_list[i][0]).cpu().numpy().transpose(1, 2, 0)[:h, :w], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
            else:
                cv2.imwrite(f'{output_dir}/img{i}.png', (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

        output_path = os.path.join(tmpdir, "output.zip")
        shutil.make_archive(output_path.split(".zip")[0], "zip", output_dir)
        return Path(output_path)

    def run_video(self, video, montage, uhd, scale, skip, fps, png, ext, exp):
        videoCapture = cv2.VideoCapture(video)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        videoCapture.release()
        if fps is None:
            fpsNotAssigned = True
            fps = fps * (2 ** exp)
        else:
            fpsNotAssigned = False
        videogen = skvideo.io.vreader(video)
        lastframe = next(videogen)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_path_wo_ext, ext = os.path.splitext(video)
        print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, ext, tot_frame, fps, fps))
        if png == False and fpsNotAssigned == True and not skip:
            print("The audio will be merged after interpolation process")
        else:
            print("Will not merge audio because using png, fps or skip flag!")

        h, w, _ = lastframe.shape
        vid_out_name = None
        vid_out = None
        if png:
            if not os.path.exists('vid_out'):
                os.mkdir('vid_out')
        else:
            vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, (2 ** exp), int(np.round(fps)), ext)
            vid_out = cv2.VideoWriter(vid_out_name, fourcc, fps, (w, h))

        if montage:
            left = w // 4
            w = w // 2
        tmp = max(32, int(32 / scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)
        skip_frame = 1
        if montage:
            lastframe = lastframe[:, left: left + w]
        write_buffer = Queue(maxsize=500)
        read_buffer = Queue(maxsize=500)

        def clear_write_buffer(write_buffer):
            cnt = 0
            while True:
                item = write_buffer.get()
                if item is None:
                    break
                if png:
                    cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
                    cnt += 1
                else:
                    vid_out.write(item[:, :, ::-1])

        def build_read_buffer(read_buffer, videogen):
            try:
                for frame in videogen:
                     if montage:
                          frame = frame[:, left: left + w]
                     read_buffer.put(frame)
            except:
                pass
            read_buffer.put(None)

        _thread.start_new_thread(build_read_buffer, (read_buffer, videogen))
        _thread.start_new_thread(clear_write_buffer, (write_buffer,))

        I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1)
        while True:
            frame = read_buffer.get()
            if frame is None:
                break
            I0 = I1
            I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
            I1 = pad_image(I1)
            I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small, I1_small)
            if ssim > 0.995 and skip:
                if skip_frame % 100 == 0:
                    print("Warning: Your video has {} static frames, skipping them may change the duration of the generated video.".format(skip_frame))
                skip_frame += 1
                continue
            if ssim < 0.5:
                output = []
                step = 1 / (2 ** exp)
                alpha = 0
                for i in range((2 ** exp) - 1):
                    alpha += step
                    beta = 1-alpha
                    output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.)
            else:
                output = make_inference(self.model, scale, I0, I1, exp)
            if montage:
                write_buffer.put(np.concatenate((lastframe, lastframe), 1))
                for mid in output:
                    mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                    write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
            else:
                write_buffer.put(lastframe)
                for mid in output:
                    mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                    write_buffer.put(mid[:h, :w])
            lastframe = frame
        if montage:
            write_buffer.put(np.concatenate((lastframe, lastframe), 1))
        else:
            write_buffer.put(lastframe)
        import time
        while(not write_buffer.empty()):
            time.sleep(0.1)
        if not vid_out is None:
            vid_out.release()

        # move audio to new video file if appropriate
        if png == False and fpsNotAssigned == True and not skip and not video is None:
            try:
                transferAudio(video, vid_out_name)
            except:
                print("Audio transfer failed. Interpolated video will have no audio")
                targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
                os.rename(targetNoAudio, vid_out_name)



def transferAudio(sourceVideo, targetVideo):
    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:

        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

    if os.path.getsize(targetVideo) == 0: # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0): # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")


def make_inference(model, scale, I0, I1, exp):
    middle = model.inference(I0, I1, scale)
    if exp == 1:
        return [middle]
    first_half = make_inference(model, scale, I0, middle, exp=exp - 1)
    second_half = make_inference(model, scale, middle, I1, exp=exp - 1)
    return [*first_half, middle, *second_half]

def pad_image(padding, img):
    return F.pad(img, padding)
