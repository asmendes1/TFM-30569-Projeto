import os
import moviepy.editor as mp
from moviepy.editor import *

class Video:
    # TODO - delete "../" from self.rel_path (this is for testing)
    def __init__(self, rel_path, file_name):
        self.rel_path = rel_path + "/" + file_name
        self.abs_path = self.get_abs_path()
        self.video_clip = self.get_videoclip(0, 0, 0, 0, 0, -1)

    def get_rel_path(self):
        return self.rel_path

    def get_abs_path(self):
        return os.path.abspath("../" + self.rel_path)

    def __get_rel_path_from_vid(self, file_name):
        return "../videos/" + file_name

    def __get_abs_path_from_vid(self, file_name):
        return os.path.abspath("../" + self.__get_rel_path_from_vid(file_name))

    def get_videoclip(self, start_hour, start_min, start_sec, end_hour, end_min, end_sec):
        vid = mp.VideoFileClip(self.rel_path).subclip((start_hour, start_min, start_sec), (end_hour, end_min, end_sec))
        return vid

    def write_videoclip(self, videoclip, vid_rel_path):
        videoclip.write_videofile(vid_rel_path)

    def get_audio_from_video(self, vid, audio_name):
        vid.audio.write_audiofile(self.rel_path + audio_name)

    def split_video(self, vd_idx_ini=0, debug=False):
        ss_ini, ss_final = 0, 59
        mm_ini, mm_final = 0, 0
        hh_ini, hh_final = 0, 0
        vd_idx = vd_idx_ini  # inicial video index thats being generated
        duration = int(self.video_clip.duration)

        for i in range(59, duration, 59):  # iterate minute by minute(59 secs)
            if debug:
                print("hora inicial ---> " + str(hh_ini) + ":" + str(mm_ini) + ":" + str(ss_ini))
                print("hora final:  ---> " + str(hh_final) + ":" + str(mm_final) + ":" + str(ss_final))

            path = "padel_" + str(vd_idx) + ".mp4"
            rel_path = self.__get_rel_path_from_vid(path)
            try:  # if it raises an exception then we reached the video limit
                vid = self.get_videoclip(hh_ini, mm_ini, ss_ini, hh_final, mm_final, ss_final)
                self.write_videoclip(vid, rel_path)
            except: break

            if mm_final == 59:
                hh_ini += 1
                hh_final += 1
                mm_ini, mm_final = 0, 0

            mm_ini += 1
            mm_final += 1
            vd_idx += 1
            if debug:
                print("===============================================")

    def get_file(self):
        return self.video_clip


def get_millis(ms):
    while ms > 999:
        ms -= 1000

    ms_value = ms
    return ms_value


def fill(value_type, value):
    if value_type != "hh" and value_type != "mm" and \
            value_type != "ss" and value_type != "ms":
        print("Wrong value type on fill method from createMiniClipsFunctions.py file")

    string = "" + str(value)
    if (value_type == "hh" or value_type == "mm" or value_type == "ss") and len(string) < 2:
        string = "0" + str(value)
    elif value_type == "ms":
        while len(string) < 3:
            string = "0" + string
    return string


def get_time(sample):
    millis = int((sample * 1000) / 44100)
    ms = get_millis(millis)
    seconds = (millis/1000) % 60
    seconds = int(seconds)
    minutes = (millis/(1000 * 60)) % 60
    minutes = int(minutes)
    hours = (millis/(1000 * 60 * 60)) % 24
    hours = hours if hours > 1 else 0  

   

    hh = fill("hh", hours)
    mm = fill("mm", minutes)
    ss = fill("ss", seconds)
    millisec = fill("ms", ms)

    s = str(hh) + ":" + str(mm) + ":" + str(ss) + "." + str(millisec)

    return s
