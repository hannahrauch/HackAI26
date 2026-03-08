import subprocess
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

from gpiozero import AngularServo
from gpiozero.pins.lgpio import LGPIOFactory
from RPLCD.i2c import CharLCD


# =========================================================
# Paths
# =========================================================
RAW_IMAGE_PATH = "/home/mym/Desktop/hackai/deploy/capture.jpg"
CROP_IMAGE_PATH = "/home/mym/Desktop/hackai/deploy/orange_640.jpg"
DETECT_MODEL_PATH = "/home/mym/Desktop/hackai/deploy/efficientdet_lite0.tflite"
CLASSIFY_MODEL_PATH = "/home/mym/Desktop/hackai/deploy/best.pt"

# =========================================================
# Settings
# =========================================================
CROP_SIZE = 640
CLASSIFY_IMGSZ = 224

# LCD
LCD_I2C_ADDR = 0x27
LCD_COLS = 16
LCD_ROWS = 2

# Servo
SERVO_BCM_PIN = 18
SERVO_HOME_ANGLE = 40
SERVO_PUSH_ANGLE = 130
SERVO_HOLD_TIME = 0.5
SERVO_RETURN_TIME = 0.7

# Safer pulse range for less jitter
SERVO_MIN_PULSE = 0.0005
SERVO_MAX_PULSE = 0.0025

# Loop timing
LOOP_DELAY = 1.0
NO_OBJECT_DELAY = 0.8
ERROR_DELAY = 1.5

# Classification labels
GOOD_LABELS = {"good", "fresh", "normal"}
BAD_LABELS = {"bad", "defect", "defective", "rotten"}

# Optional confidence threshold
MIN_CONFIDENCE = 0.0

# =========================================================
# Global counters
# =========================================================
good_count = 0
bad_count = 0


# =========================================================
# LCD functions
# =========================================================
def init_lcd():
    lcd = CharLCD(
        i2c_expander="PCF8574",
        address=LCD_I2C_ADDR,
        port=1,
        cols=LCD_COLS,
        rows=LCD_ROWS,
        dotsize=8,
        charmap="A02",
        auto_linebreaks=False,
    )
    lcd.clear()
    return lcd


def lcd_write_2lines(lcd, line1="", line2=""):
    lcd.clear()
    lcd.cursor_pos = (0, 0)
    lcd.write_string(line1[:LCD_COLS].ljust(LCD_COLS))
    lcd.cursor_pos = (1, 0)
    lcd.write_string(line2[:LCD_COLS].ljust(LCD_COLS))


def update_counter_display(lcd, status="READY"):
    line1 = f"G:{good_count} B:{bad_count}"
    line2 = status
    lcd_write_2lines(lcd, line1, line2)


# =========================================================
# Servo functions
# =========================================================
def init_servo():
    factory = LGPIOFactory()
    servo = AngularServo(
        SERVO_BCM_PIN,
        min_angle=0,
        max_angle=180,
        initial_angle=SERVO_HOME_ANGLE,
        min_pulse_width=SERVO_MIN_PULSE,
        max_pulse_width=SERVO_MAX_PULSE,
        pin_factory=factory,
    )
    servo.detach()  
    time.sleep(0.5)
    return servo


def sweep_once_and_return(servo):
    servo.angle = SERVO_PUSH_ANGLE
    time.sleep(SERVO_HOLD_TIME)
    servo.angle = SERVO_HOME_ANGLE
    time.sleep(SERVO_RETURN_TIME)
    servo.detach() 


# =========================================================
# Camera capture
# =========================================================
def capture_image(output_path: str):
    cmd = [
        "rpicam-still",
        "-n",
        "-o", output_path,
        "--width", "1080",
        "--height", "1080"
    ]
    subprocess.run(cmd, check=True)
    print(f"Captured: {output_path}")


# =========================================================
# Crop helpers
# =========================================================
def make_square_crop(img, x, y, w, h, out_size=640):
    ih, iw = img.shape[:2]

    cx = x + w // 2
    cy = y + h // 2

    side = max(w, h)
    side = int(side * 1.2)
    half = side // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(iw, cx + half)
    y2 = min(ih, cy + half)

    crop = img[y1:y2, x1:x2]

    ch, cw = crop.shape[:2]
    if ch != cw:
        side2 = max(ch, cw)
        padded = cv2.copyMakeBorder(
            crop,
            top=(side2 - ch) // 2,
            bottom=side2 - ch - (side2 - ch) // 2,
            left=(side2 - cw) // 2,
            right=side2 - cw - (side2 - cw) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        crop = padded

    crop_640 = cv2.resize(crop, (out_size, out_size))
    return crop_640


# =========================================================
# Model init
# =========================================================
def init_detector():
    base_options = python.BaseOptions(model_asset_path=DETECT_MODEL_PATH)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=0.3,
        max_results=5
    )
    return vision.ObjectDetector.create_from_options(options)


def init_classifier():
    return YOLO(CLASSIFY_MODEL_PATH)


# =========================================================
# Detection and classification
# =========================================================
def detect_and_crop(detector, input_path: str, output_path: str):
    mp_image = mp.Image.create_from_file(input_path)
    img = cv2.imread(input_path)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    result = detector.detect(mp_image)

    if not result.detections:
        raise RuntimeError("No orange detected.")

    best_det = None
    best_area = -1

    for detection in result.detections:
        bbox = detection.bounding_box
        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
        area = w * h
        if area > best_area:
            best_area = area
            best_det = bbox

    x, y, w, h = best_det.origin_x, best_det.origin_y, best_det.width, best_det.height

    crop_640 = make_square_crop(img, x, y, w, h, out_size=CROP_SIZE)
    cv2.imwrite(output_path, crop_640)

    print(f"Cropped and saved: {output_path}")
    print(f"Bounding box: x={x}, y={y}, w={w}, h={h}")


def classify_image(model, image_path: str):
    result = model.predict(source=image_path, imgsz=CLASSIFY_IMGSZ, verbose=False)[0]

    if result.probs is None:
        raise RuntimeError("Classifier returned no probability output.")

    label = result.names[int(result.probs.top1)].strip().lower()
    conf = float(result.probs.top1conf)

    print(f"Prediction: {label}")
    print(f"Confidence: {conf:.4f}")
    print(f"FINAL: {label.upper()}")

    return label, conf


def is_bad_label(label: str):
    return label in BAD_LABELS


def is_good_label(label: str):
    return label in GOOD_LABELS


# =========================================================
# Main loop
# =========================================================
def main():
    global good_count, bad_count

    if not Path(DETECT_MODEL_PATH).exists():
        raise FileNotFoundError(f"Missing detector model: {DETECT_MODEL_PATH}")

    if not Path(CLASSIFY_MODEL_PATH).exists():
        raise FileNotFoundError(f"Missing classifier model: {CLASSIFY_MODEL_PATH}")

    lcd = init_lcd()
    servo = init_servo()
    detector = init_detector()
    classifier = init_classifier()

    update_counter_display(lcd, "STARTING")
    time.sleep(1)

    try:
        while True:
            try:
                update_counter_display(lcd, "CAPTURE")
                capture_image(RAW_IMAGE_PATH)

                update_counter_display(lcd, "DETECT")
                detect_and_crop(detector, RAW_IMAGE_PATH, CROP_IMAGE_PATH)

                update_counter_display(lcd, "CLASSIFY")
                label, conf = classify_image(classifier, CROP_IMAGE_PATH)

                if conf < MIN_CONFIDENCE:
                    update_counter_display(lcd, "LOW CONF")
                    time.sleep(LOOP_DELAY)
                    continue

                if is_bad_label(label):
                    bad_count += 1
                    update_counter_display(lcd, f"BAD {conf:.2f}")

                    # Re‑attach and sweep
                    servo.angle = SERVO_PUSH_ANGLE    # attaches and moves to push
                    time.sleep(SERVO_HOLD_TIME)
                    servo.angle = SERVO_HOME_ANGLE    # moves back home
                    time.sleep(SERVO_RETURN_TIME)
                    servo.detach()                     # detach again to avoid jitter

                elif is_good_label(label):
                    good_count += 1
                    update_counter_display(lcd, f"GOOD {conf:.2f}")

                else:
                    update_counter_display(lcd, label[:16])
                    print(f"Unknown label '{label}', no servo action.")

                time.sleep(LOOP_DELAY)

            except RuntimeError as e:
                print(f"[WARN] {e}")
                update_counter_display(lcd, "NO ORANGE")
                time.sleep(NO_OBJECT_DELAY)

            except Exception as e:
                print(f"[ERROR] {e}")
                update_counter_display(lcd, "ERROR")
                time.sleep(ERROR_DELAY)

    except KeyboardInterrupt:
        print("Stopped by user.")

    finally:
        try:
            servo.angle = SERVO_HOME_ANGLE
        except Exception:
            pass

        try:
            lcd_write_2lines(lcd, "Program stopped", f"G:{good_count} B:{bad_count}")
        except Exception:
            pass


if __name__ == "__main__":
    main()