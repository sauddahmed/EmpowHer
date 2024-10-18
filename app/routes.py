from flask import render_template, Response, request
from app import app
import cv2
import math
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
)


exercise_stats = {
    "bicep_curls": {"reps_left": 0, "reps_right": 0, "time": 0},
    "pushups": {"reps": 0, "time": 0},
    "squats": {"reps": 0, "time": 0},
    "lunges": {"reps_left": 0, "reps_right": 0, "time": 0},
    "shoulderPress": {"reps": 0, "time": 0},
    "crunches": {"reps": 0, "time": 0},
    "mountainClimbers": {"reps": 0, "time": 0},
}


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def gen_bicepCurl_frames():
    cap = cv2.VideoCapture(0)
    counter_left = 0
    counter_right = 0
    stage_left = None
    stage_right = None

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            ]
            left_elbow = [
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
            ]
            left_wrist = [
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
            ]

            right_shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
            ]
            right_elbow = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
            ]
            right_wrist = [
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
            ]

            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            if left_angle > 160:
                stage_left = "down"
            if left_angle < 30 and stage_left == "down":
                stage_left = "up"
                counter_left += 1

            if right_angle > 160:
                stage_right = "down"
            if right_angle < 30 and stage_right == "down":
                stage_right = "up"
                counter_right += 1

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            timer_text = f"Time: {minutes:02}:{seconds:02}"

            # Update exercise statistics
            exercise_stats["bicep_curls"]["reps_left"] = counter_left
            exercise_stats["bicep_curls"]["reps_right"] = counter_right
            exercise_stats["bicep_curls"]["time"] = elapsed_time

            cv2.putText(
                image,
                "Left Reps: " + str(counter_left),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                "Right Reps: " + str(counter_right),
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 140, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                timer_text,
                (430, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        except:
            pass

        ret, buffer = cv2.imencode(".jpg", image)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()
    cv2.destroyAllWindows()


def gen_pushup_frames():
    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            ]
            left_elbow = [
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
            ]
            left_wrist = [
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
            ]

            right_shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
            ]
            right_elbow = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
            ]
            right_wrist = [
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
            ]

            # Calculate angles for both arms
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Use the average angle to determine push-up stage
            angle = (left_angle + right_angle) / 2

            if angle > 160:
                stage = "down"
            if angle < 90 and stage == "down":
                stage = "up"
                counter += 1

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            timer_text = f"Time: {minutes:02}:{seconds:02}"

            # Update exercise statistics
            exercise_stats["pushups"]["reps"] = counter
            exercise_stats["pushups"]["time"] = elapsed_time

            cv2.putText(
                image,
                "Reps: " + str(counter),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                timer_text,
                (430, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        except Exception as e:
            print(f"Error processing frame: {e}")
            pass

        ret, buffer = cv2.imencode(".jpg", image)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()
    cv2.destroyAllWindows()


def gen_squat_frames():
    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
            ]
            knee = [
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
            ]
            ankle = [
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
            ]

            angle = calculate_angle(hip, knee, ankle)

            if angle > 170:
                stage = "up"
            if angle < 90 and stage == "up":
                stage = "down"
                counter += 1

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            timer_text = f"Time: {minutes:02}:{seconds:02}"

            # Update exercise statistics
            exercise_stats["squats"]["reps"] = counter
            exercise_stats["squats"]["time"] = elapsed_time

            cv2.putText(
                image,
                "Reps: " + str(counter),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                timer_text,
                (430, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        except:
            pass

        ret, buffer = cv2.imencode(".jpg", image)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()
    cv2.destroyAllWindows()


def gen_lunge_frames():
    cap = cv2.VideoCapture(0)
    counter_left = 0
    counter_right = 0
    stage_left = None
    stage_right = None

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            left_hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
            ]
            left_knee = [
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
            ]
            left_ankle = [
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
            ]

            right_hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
            ]
            right_knee = [
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
            ]
            right_ankle = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
            ]

            left_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_angle = calculate_angle(right_hip, right_knee, right_ankle)

            if left_angle > 160:
                stage_left = "up"
            if left_angle < 90 and stage_left == "up":
                stage_left = "down"
                counter_left += 1

            if right_angle > 160:
                stage_right = "up"
            if right_angle < 90 and stage_right == "up":
                stage_right = "down"
                counter_right += 1

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            timer_text = f"Time: {minutes:02}:{seconds:02}"

            # Update exercise statistics
            exercise_stats["lunges"]["reps_left"] = counter_left
            exercise_stats["lunges"]["reps_right"] = counter_right
            exercise_stats["lunges"]["time"] = elapsed_time

            cv2.putText(
                image,
                "Left Reps: " + str(counter_left),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                "Right Reps: " + str(counter_right),
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 140, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                timer_text,
                (430, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        except:
            pass

        ret, buffer = cv2.imencode(".jpg", image)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()
    cv2.destroyAllWindows()


def gen_shoulderPress_frames():
    cap = cv2.VideoCapture(0)
    counter = 0
    stage = "down"

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for left and right elbows, shoulders, and wrists
            left_shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            ]
            left_elbow = [
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
            ]
            left_wrist = [
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
            ]

            right_shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
            ]
            right_elbow = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
            ]
            right_wrist = [
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
            ]

            # Calculate angles
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(
                right_shoulder, right_elbow, right_wrist
            )

            # Use the average angle to determine push-up stage
            angle = (left_elbow_angle + right_elbow_angle) / 2

            # Shoulder press logic
            if angle > 160 and stage == "down":
                stage = "up"
                counter += 1
            if angle < 90:
                stage = "down"

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            timer_text = f"Time: {minutes:02}:{seconds:02}"

            # Update exercise statistics
            exercise_stats["shoulderPress"]["reps"] = counter
            exercise_stats["shoulderPress"]["time"] = elapsed_time

            # Render results
            cv2.putText(
                image,
                "Reps: " + str(counter),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                timer_text,
                (430, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        except Exception as e:
            print(f"Error: {e}")
            pass

        ret, buffer = cv2.imencode(".jpg", image)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()
    cv2.destroyAllWindows()


def gen_crunches_frames():
    cap = cv2.VideoCapture(0)
    counter = 0
    stage = "down"

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for left and right hips, shoulders, elbows, and knees
            left_hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
            ]
            left_elbow = [
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
            ]
            left_knee = [
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
            ]

            right_hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
            ]
            right_elbow = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
            ]
            right_knee = [
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
            ]

            # Calculate distances
            left_distance = calculate_distance(left_elbow, left_knee)
            right_distance = calculate_distance(right_elbow, right_knee)

            # Use the average distance to determine crunch stage
            distance = (left_distance + right_distance) / 2

            # Crunch logic
            if distance < 0.15 and stage == "down":
                stage = "up"
                counter += 1
                print(f"Rep {counter} counted.")
            if distance > 0.30:
                stage = "down"

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            timer_text = f"Time: {minutes:02}:{seconds:02}"

            # Update exercise statistics
            exercise_stats["crunches"]["reps"] = counter
            exercise_stats["crunches"]["time"] = elapsed_time

            # Render results
            cv2.putText(
                image,
                "Reps: " + str(counter),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                timer_text,
                (430, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        except Exception as e:
            print(f"Error: {e}")
            pass

        ret, buffer = cv2.imencode(".jpg", image)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()
    cv2.destroyAllWindows()


def gen_mountainClimbers_frames():
    counter = 0
    cap = cv2.VideoCapture(0)
    stage = None
    threshold_distance = 0.1  # Adjust this value based on your camera setup

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for left and right knees and hips
            left_hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
            ]
            left_knee = [
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
            ]

            right_hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
            ]
            right_knee = [
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
            ]

            # Calculate distances
            left_distance = calculate_distance(left_hip, left_knee)
            right_distance = calculate_distance(right_hip, right_knee)

            # Mountain climbers logic
            if (
                left_distance < threshold_distance
                or right_distance < threshold_distance
            ):
                if stage != "up":
                    stage = "up"
            else:
                if stage == "up":
                    stage = "down"
                    counter += 1
                    print(f"Rep {counter} counted.")

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            timer_text = f"Time: {minutes:02}:{seconds:02}"

            # Update exercise statistics
            exercise_stats["mountainClimbers"]["reps"] = counter
            exercise_stats["mountainClimbers"]["time"] = elapsed_time

            # Render results
            cv2.putText(
                image,
                "Reps: " + str(counter),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                timer_text,
                (430, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        except Exception as e:
            print(f"Error: {e}")
            pass

        ret, buffer = cv2.imencode(".jpg", image)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()
    cv2.destroyAllWindows()

    # Function to calculate calories


def calculate_calories(weight, height, age, gender, activity_level):
    # Basal Metabolic Rate (BMR) calculation
    if gender == "male":
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

    # Activity level multiplier
    if activity_level == "sedentary":
        calories_needed = bmr * 1.2
    elif activity_level == "light":
        calories_needed = bmr * 1.375
    elif activity_level == "moderate":
        calories_needed = bmr * 1.55
    elif activity_level == "active":
        calories_needed = bmr * 1.725
    else:
        calories_needed = bmr * 1.9

    return {
        "maintenance": round(calories_needed),
        "bulking": round(calories_needed + 500),
        "cutting": round(calories_needed - 500),
    }


@app.route("/pushups")
def pushups():
    return render_template("pushups.html")


@app.route("/pushup_video_feed")
def pushup_video_feed():
    return Response(
        gen_pushup_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/BicepCurls")
def BicepCurls():
    return render_template("BicepCurls.html")


@app.route("/BicepCurls_video_feed")
def BicepCurls_video_feed():
    return Response(
        gen_bicepCurl_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/squats")
def squats():
    return render_template("squats.html")


@app.route("/squat_video_feed")
def squat_video_feed():
    return Response(
        gen_squat_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/lunges")
def lunges():
    return render_template("lunges.html")


@app.route("/lunge_video_feed")
def lunge_video_feed():
    return Response(
        gen_lunge_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/shoulderPress")
def shoulderPress():
    return render_template("shoulderPress.html")


@app.route("/shoulderPress_video_feed")
def shoulderPress_video_feed():
    return Response(
        gen_shoulderPress_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/crunches")
def crunches():
    return render_template("crunches.html")


@app.route("/crunches_video_feed")
def crunches_video_feed():
    return Response(
        gen_crunches_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/mountainClimbers")
def mountainClimbers():
    return render_template("mountainClimbers.html")


@app.route("/mountainClimbers_video_feed")
def mountainClimbers_video_feed():
    return Response(
        gen_mountainClimbers_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/bicep_curls_statistics")
def bicep_curls_statistics():
    return render_template(
        "bicep_curls_statistics.html", stats=exercise_stats["bicep_curls"]
    )


@app.route("/pushups_statistics")
def pushups_statistics():
    return render_template("pushups_statistics.html", stats=exercise_stats["pushups"])


@app.route("/squats_statistics")
def squats_statistics():
    return render_template("squats_statistics.html", stats=exercise_stats["squats"])


@app.route("/lunges_statistics")
def lunges_statistics():
    return render_template("lunges_statistics.html", stats=exercise_stats["lunges"])


@app.route("/shoulderPress_statistics")
def shoulderPress_statistics():
    return render_template(
        "shoulderPress_statistics.html", stats=exercise_stats["shoulderPress"]
    )


@app.route("/crunches_statistics")
def crunches_statistics():
    return render_template("crunches_statistics.html", stats=exercise_stats["crunches"])


@app.route("/mountainClimbers_statistics")
def mountainClimbers_statistics():
    return render_template(
        "mountainClimbers_statistics.html", stats=exercise_stats["mountainClimbers"]
    )


@app.route("/BicepCurls_LM")
def BicepCurls_LM():
    return render_template("BicepCurls_LM.html")


@app.route("/pushups_LM")
def pushups_LM():
    return render_template("pushups_LM.html")


@app.route("/squats_LM")
def squats_LM():
    return render_template("squats_LM.html")


@app.route("/lunges_LM")
def lunges_LM():
    return render_template("lunges_LM.html")


@app.route("/shoulderPress_LM")
def shoulderPress_LM():
    return render_template("shoulderPress_LM.html")


@app.route("/crunches_LM")
def crunches_LM():
    return render_template("crunches_LM.html")


@app.route("/mountainClimbers_LM")
def mountainClimbers_LM():
    return render_template("mountainClimbers_LM.html")


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/bmi_calculator")
def bmi_calculator():
    return render_template("bmi_calculator.html")


@app.route("/periods")
def periods():
    return render_template("periods.html")


@app.route("/calculate_bmi", methods=["POST"])
def calculate_bmi():
    weight = float(request.form["weight"])
    height_feet = float(request.form["height_feet"])
    height_inches = float(request.form["height_inches"])
    total_inches = (height_feet * 12) + height_inches
    height_cm = total_inches * 2.54  # Convert inches to cm
    height_meters = height_cm / 100  # Convert cm to meters
    bmi = weight / (height_meters**2)
    return render_template("bmi_calculator.html", bmi=round(bmi, 2))


@app.route("/calorie_calculator", methods=["GET", "POST"])
def calorie_calculator():
    if request.method == "POST":
        age = int(request.form["age"])
        gender = request.form["gender"]
        height_feet = float(request.form["height_feet"])
        height_inches = float(request.form["height_inches"])

        height = (height_feet * 30.48) + (height_inches * 2.54)
        weight = float(request.form["weight"])
        activity_level = request.form["activity_level"]
        calories = calculate_calories(weight, height, age, gender, activity_level)
        return render_template("calorie_calculator.html", calories=calories)
    return render_template("calorie_calculator.html", calories=None)
