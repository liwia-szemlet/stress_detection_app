import os
import sys
import cv2
import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFormLayout, QMessageBox,
    QStackedWidget, QSizePolicy, QTextEdit
)

from face_tracker_for_app import FaceTracker  # nasz tracker

# singleton trackera
_tracker = FaceTracker(user_id="demo_user")


def process_frame(frame):
    """
    Wywoływane w wątku kamery.
    """
    return _tracker.process(frame)


# ---------- CAMERA THREAD (VIDEO + LOGI) ----------

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    log_ready = pyqtSignal(str)

    def __init__(self, parent=None, camera_index=0):
        super().__init__(parent)
        self._run_flag = True
        self.camera_index = camera_index

    def run(self):
        self.log_ready.emit("Opening camera...\n")
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.log_ready.emit("ERROR: Cannot open camera.\n")
            self._run_flag = False

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                self.log_ready.emit("WARNING: Failed to read frame from camera.\n")
                break

            try:
                processed = process_frame(frame)

                # LOGI Z FEATUREÓW
                feats = _tracker.get_features()
                if feats:
                    self.log_ready.emit(
                        f"{feats['timestamp']:.2f} "
                        f"idx={feats.get('stress_index_0100', 0.0):.1f} "
                        f"brow={feats.get('brow_mean', 0.0):.3f} "
                        f"mouth={feats.get('mouth_mean', 0.0):.3f} "
                        f"eye={feats.get('eye_mean', 0.0):.3f} "
                        f"emo={feats.get('emotion_label', 'neutral')}\n"
                    )

            except Exception as e:
                self.log_ready.emit(f"ERROR in process_frame: {e}\n")
                processed = frame

            self.frame_ready.emit(processed)

        cap.release()
        self.log_ready.emit("Camera released.\n")

    def stop(self):
        self._run_flag = False
        self.wait()


# ---------- SCREEN 1: SPLASH / START ----------

class SplashScreen(QWidget):
    def __init__(self, on_start_callback, parent=None):
        super().__init__(parent)
        self.on_start_callback = on_start_callback

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        logo_label = QLabel("LOGO")
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setStyleSheet("font-size: 32px; font-weight: bold;")
        layout.addWidget(logo_label)

        name_label = QLabel("MoodAgent")
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet("font-size: 24px;")
        layout.addWidget(name_label)

        start_button = QPushButton("Start application")
        start_button.setFixedWidth(200)
        start_button.clicked.connect(self.on_start_callback)
        layout.addWidget(start_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)


# ---------- SCREEN 2: LOGIN ----------

class LoginScreen(QWidget):
    def __init__(self, on_login_callback, parent=None):
        super().__init__(parent)
        self.on_login_callback = on_login_callback

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        title = QLabel("Please log in")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px;")
        main_layout.addWidget(title)

        form = QFormLayout()
        self.user_edit = QLineEdit()
        self.pass_edit = QLineEdit()
        self.pass_edit.setEchoMode(QLineEdit.Password)

        form.addRow("Username:", self.user_edit)
        form.addRow("Password:", self.pass_edit)

        form_container = QWidget()
        form_container.setLayout(form)
        form_container.setMaximumWidth(300)
        main_layout.addWidget(form_container, alignment=Qt.AlignCenter)

        forgot_label = QLabel('<a href="#">Forgot your password?</a>')
        forgot_label.setOpenExternalLinks(False)
        forgot_label.linkActivated.connect(self.on_forgot_clicked)
        forgot_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(forgot_label)

        login_button = QPushButton("Log me in")
        login_button.setFixedWidth(200)
        login_button.clicked.connect(self.on_login_clicked)
        main_layout.addWidget(login_button, alignment=Qt.AlignCenter)

        self.setLayout(main_layout)

    def on_forgot_clicked(self):
        QMessageBox.information(
            self, "Forgot password",
            "Password recovery is not available in this demo version."
        )

    def on_login_clicked(self):
        username = self.user_edit.text().strip()
        if not username:
            username = "anonymous"

        QMessageBox.information(self, "Login", f"Logged in as {username}")
        self.on_login_callback(username)


# ---------- TAB 1: FACE ACQUISITION ----------

class FaceTrackerTab(QWidget):
    def __init__(self, logs_tab=None, parent=None):
        super().__init__(parent)

        self.camera_thread = None
        self.logs_tab = logs_tab

        layout = QVBoxLayout()

        self.status_label = QLabel("Status: stopped")
        layout.addWidget(self.status_label)

        self.video_placeholder = QLabel("Not available")
        self.video_placeholder.setAlignment(Qt.AlignCenter)
        self.video_placeholder.setStyleSheet(
            "background-color: #222; color: #ccc; padding: 20px;"
        )
        self.video_placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_placeholder.setMinimumSize(640, 480)
        layout.addWidget(self.video_placeholder)

        btn_layout = QHBoxLayout()
        self.start_button = QPushButton("Start acquisition")
        self.stop_button = QPushButton("Stop acquisition")
        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_tracker)
        self.stop_button.clicked.connect(self.stop_tracker)

        btn_layout.addWidget(self.start_button)
        btn_layout.addWidget(self.stop_button)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def start_tracker(self):
        if self.camera_thread is None:
            self.camera_thread = CameraThread()
            self.camera_thread.frame_ready.connect(self.update_frame)

            if self.logs_tab is not None:
                self.camera_thread.log_ready.connect(self.logs_tab.append_log)

            self.camera_thread.start()

            self.status_label.setText("Status: running")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

            QMessageBox.information(
                self, "Info",
                "Face expressions acquisition started."
            )
        else:
            QMessageBox.information(self, "Info", "Face tracker is already running.")

    def stop_tracker(self):
        if self.camera_thread is not None:
            if self.logs_tab is not None:
                try:
                    self.camera_thread.log_ready.disconnect(self.logs_tab.append_log)
                except TypeError:
                    pass
            self.camera_thread.stop()
            self.camera_thread = None

        self.status_label.setText("Status: stopped")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        self.video_placeholder.clear()
        self.video_placeholder.setText("Not available")

        if self.logs_tab is not None:
            self.logs_tab.append_log("\n--- session stopped ---\n")

        QMessageBox.information(
            self, "Info",
            "Face acquisition stopped."
        )

    def emergency_kill(self):
        if self.camera_thread is not None:
            if self.logs_tab is not None:
                try:
                    self.camera_thread.log_ready.disconnect(self.logs_tab.append_log)
                except TypeError:
                    pass
            self.camera_thread.stop()
            self.camera_thread = None

        self.video_placeholder.clear()
        self.video_placeholder.setText("Not available")
        if self.logs_tab is not None:
            self.logs_tab.append_log("\n--- session stopped (emergency) ---\n")

    @pyqtSlot(np.ndarray)
    def update_frame(self, frame: np.ndarray):
        if self.status_label.text() != "Status: running":
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        target_w = self.video_placeholder.width()
        target_h = self.video_placeholder.height()
        scaled = image.scaled(
            target_w,
            target_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_placeholder.setPixmap(QPixmap.fromImage(scaled))
        if self.video_placeholder.text():
            self.video_placeholder.setText("")


# ---------- TAB 2: REAL-TIME ANALYTICS (placeholder) ----------

class AnalyticsTab(QWidget):
    def __init__(self, quicksight_url: str, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        label = QLabel(
            "Real-time analytics\n\n"
            "QuickSight dashboard placeholder.\n"
            "Web view can be added later."
        )
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        layout.addWidget(label)
        self.setLayout(layout)


# ---------- TAB 3: REAL-TIME LOGS ----------

class LogsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        header_layout = QHBoxLayout()
        header_layout.addStretch()
        self.erase_button = QPushButton("Erase old logs")
        header_layout.addWidget(self.erase_button)
        layout.addLayout(header_layout)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("background-color: #111; color: #0f0;")
        layout.addWidget(self.text_edit)

        self.setLayout(layout)

        self.erase_button.clicked.connect(self.clear_logs)

    @pyqtSlot(str)
    def append_log(self, message: str):
        cursor = self.text_edit.textCursor()
        cursor.movePosition(cursor.End)
        self.text_edit.setTextCursor(cursor)
        self.text_edit.insertPlainText(message)

    def clear_logs(self):
        self.text_edit.clear()
        self.append_log("--- logs cleared ---\n")


# ---------- TAB 4: SETTINGS & PRIVACY ----------

class SettingsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        title = QLabel("Settings and Privacy")
        title.setStyleSheet("font-size: 18px;")
        layout.addWidget(title)

        info = QLabel(
            "Here you can manage data upload settings, consent, and privacy policies."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addStretch()
        self.setLayout(layout)


# ---------- MAIN APP WINDOW ----------

class MainAppWindow(QWidget):
    def __init__(self, quicksight_url: str, username: str, parent=None):
        super().__init__(parent)

        self.username = username

        main_layout = QHBoxLayout()
        side_layout = QVBoxLayout()

        self.logs_tab = LogsTab()
        self.face_tab = FaceTrackerTab(logs_tab=self.logs_tab)
        self.analytics_tab = AnalyticsTab(quicksight_url)
        self.settings_tab = SettingsTab()

        self.btn_face = QPushButton("Face acquisition")
        self.btn_analytics = QPushButton("Real-time analytics")
        self.btn_logs = QPushButton("Real-time logs")
        self.btn_settings = QPushButton("Settings & privacy")

        for btn in (self.btn_face, self.btn_analytics, self.btn_logs, self.btn_settings):
            btn.setCheckable(True)
            btn.setMinimumHeight(40)
            btn.setStyleSheet("text-align: left; padding-left: 10px;")

        self.btn_face.setChecked(True)

        side_layout.addWidget(self.btn_face)
        side_layout.addWidget(self.btn_analytics)
        side_layout.addWidget(self.btn_logs)
        side_layout.addWidget(self.btn_settings)
        side_layout.addStretch()

        self.stack = QStackedWidget()
        self.stack.addWidget(self.face_tab)       # 0
        self.stack.addWidget(self.analytics_tab)  # 1
        self.stack.addWidget(self.logs_tab)       # 2
        self.stack.addWidget(self.settings_tab)   # 3

        self.btn_face.clicked.connect(lambda: self.set_page(0))
        self.btn_analytics.clicked.connect(lambda: self.set_page(1))
        self.btn_logs.clicked.connect(lambda: self.set_page(2))
        self.btn_settings.clicked.connect(lambda: self.set_page(3))

        main_layout.addLayout(side_layout, 0)
        main_layout.addWidget(self.stack, 1)
        self.setLayout(main_layout)

    def set_page(self, index: int):
        self.stack.setCurrentIndex(index)
        self.btn_face.setChecked(index == 0)
        self.btn_analytics.setChecked(index == 1)
        self.btn_logs.setChecked(index == 2)
        self.btn_settings.setChecked(index == 3)

    def emergency_shutdown(self):
        self.face_tab.emergency_kill()
        _tracker.close()


# ---------- MAIN WINDOW + EMERGENCY SHUTDOWN ----------

class MainWindow(QMainWindow):
    def __init__(self, quicksight_url: str):
        super().__init__()

        self.setWindowTitle("MoodAgent")
        self.resize(1100, 750)

        self.quicksight_url = quicksight_url

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.main_app_widget = None

        self.splash = SplashScreen(on_start_callback=self.show_login)
        self.stack.addWidget(self.splash)

        self.login = LoginScreen(on_login_callback=self.on_logged_in)
        self.stack.addWidget(self.login)

        self.init_emergency_button()

    def init_emergency_button(self):
        emergency_btn = QPushButton("●")
        emergency_btn.setToolTip("Emergency shutdown")
        emergency_btn.setStyleSheet(
            """
            QPushButton {
                color: red;
                font-size: 18px;
                border: none;
                background: transparent;
            }
            QPushButton::hover {
                color: #ff6666;
            }
            """
        )
        emergency_btn.setFixedSize(24, 24)
        emergency_btn.clicked.connect(self.handle_emergency_shutdown)

        toolbar = self.addToolBar("Emergency")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setStyleSheet("QToolBar { border: 0px; }")

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        toolbar.addWidget(emergency_btn)

    def show_login(self):
        self.stack.setCurrentWidget(self.login)

    def on_logged_in(self, username: str):
        self.main_app_widget = MainAppWindow(self.quicksight_url, username)
        self.stack.addWidget(self.main_app_widget)
        self.stack.setCurrentWidget(self.main_app_widget)

    def handle_emergency_shutdown(self):
        if self.main_app_widget is not None:
            self.main_app_widget.emergency_shutdown()
        QApplication.quit()

    def closeEvent(self, event):
        if self.main_app_widget is not None:
            self.main_app_widget.emergency_shutdown()
        super().closeEvent(event)


def main():
    quicksight_url = os.environ.get("QUICKSIGHT_EMBED_URL", "")
    app = QApplication(sys.argv)
    window = MainWindow(quicksight_url)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
