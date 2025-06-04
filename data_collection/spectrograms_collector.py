#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import signal
import os
import csv
from datetime import datetime
from gnuradio import gr, qtgui, uhd
from gnuradio.fft import window
from PyQt5 import Qt, QtCore
import sip
from config import SPECTROGRAM_FOLDER, SPECTROGRAMS_SAMPLES, FFT_SIZE


class WaterfallCapture(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "High-Resolution Waterfall Capture")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Waterfall Spectrum Analyzer")
        self.setLayout(Qt.QVBoxLayout())

        ##################################################
        # Parameters
        ##################################################
        self.samp_rate = 25e6
        self.center_freq = 2.437e9
        self.gain = 76
        self.antenna = "RX2"
        self.usrp_serial = "3111827"
        self.output_folder = SPECTROGRAM_FOLDER
        self.max_images = SPECTROGRAMS_SAMPLES  
        self.today_str = time.strftime("%Y%m%d")

        # Ensure the output directory exists
        os.makedirs(self.output_folder, exist_ok=True)

        ##################################################
        # USRP Source
        ##################################################
        self.source = uhd.usrp_source(
            ",".join((f"serial={self.usrp_serial}", "")),
            uhd.stream_args(cpu_format="fc32", channels=[0])
        )
        self.source.set_samp_rate(self.samp_rate)
        self.source.set_center_freq(self.center_freq, 0)
        self.source.set_gain(self.gain, 0)
        self.source.set_antenna(self.antenna, 0)

        ##################################################
        # Waterfall Sink (High-Resolution, No Title)
        ##################################################
        self.waterfall = qtgui.waterfall_sink_c(
            FFT_SIZE, window.WIN_BLACKMAN_hARRIS,
            self.center_freq, self.samp_rate,
            "",  # No title
            1
        )
        self.waterfall.enable_grid(True)
        self.waterfall.set_intensity_range(-140, 10)
        self.waterfall.set_time_per_fft(0.01)

        widget = self.waterfall.qwidget()
        if widget:
            self.waterfall_win = sip.wrapinstance(widget, Qt.QWidget)
            self.layout().addWidget(self.waterfall_win)

        ##################################################
        # Screenshot Counter: start from last existing for today
        ##################################################
        self.screenshot_counter = self.get_last_screenshot_number()

        ##################################################
        # Open CSV for logging capture times
        ##################################################
        self.csv_file_path = os.path.join(self.output_folder, "time_info.csv")
        self.csv_file = open(self.csv_file_path, mode='a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if os.stat(self.csv_file_path).st_size == 0:
            self.csv_writer.writerow(["Image Number", "Time"])  

        ##################################################
        # Connections
        ##################################################
        self.connect(self.source, self.waterfall)

        ##################################################
        # Start periodic capture after 20 seconds
        ##################################################
        QtCore.QTimer.singleShot(20000, self.start_capture_timer)

    def get_last_screenshot_number(self):
        max_number = 0
        for fname in os.listdir(self.output_folder):
            if fname.startswith(self.today_str) and fname.endswith(".png"):
                try:
                    number = int(fname.split("_")[1].split(".")[0])
                    max_number = max(max_number, number)
                except (IndexError, ValueError):
                    continue
        return max_number

    def start_capture_timer(self):
        self.capture_timer = QtCore.QTimer()
        self.capture_timer.timeout.connect(self.capture_spectrum_image)
        self.capture_timer.start(50)  # Every 0.05 s
        print("[INFO] Started periodic image capture every 0.05s after 20s delay.")

    def capture_spectrum_image(self):
        self.screenshot_counter += 1
        filename = f"{self.today_str}_{self.screenshot_counter}.png"
        filepath = os.path.join(self.output_folder, filename)
        pixmap = self.waterfall_win.grab()
        pixmap.save(filepath)

        # Log timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Up to milliseconds
        self.csv_writer.writerow([self.screenshot_counter, datetime.now().strftime('%H:%M:%S.%f')[:-3]])
        self.csv_file.flush()  # Save immediately

        print(f"[INFO] Saved: {filepath} at {timestamp}")

        # === Check if max_images is reached ===
        if self.screenshot_counter >= self.max_images:
            print(f"[INFO] Reached {self.max_images} images. Stopping capture.")
            self.capture_timer.stop()
            self.close()

    def closeEvent(self, event):
        self.capture_timer.stop()
        self.csv_file.close()
        self.stop()
        self.wait()
        event.accept()

def main():
    qapp = Qt.QApplication(sys.argv)
    tb = WaterfallCapture()
    tb.start()
    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    qapp.exec_()

if __name__ == '__main__':
    main()
