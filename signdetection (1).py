{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Import and Install Dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python mediapipe sklearn matplotlib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "import os\n",
    "from matplotlib import pyplot as plt\n",
    "import time\n",
    "import mediapipe as mp"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Keypoints using MP Holistic"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "mp_holistic = mp.solutions.holistic # Holistic model\n",
    "mp_drawing = mp.solutions.drawing_utils # Drawing utilities"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def mediapipe_detection(image, model):\n",
    "    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB\n",
    "    image.flags.writeable = False                  # Image is no longer writeable\n",
    "    results = model.process(image)                 # Make prediction\n",
    "    image.flags.writeable = True                   # Image is now writeable \n",
    "    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR\n",
    "    return image, results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def draw_landmarks(image, results):\n",
    "    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections\n",
    "    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections\n",
    "    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections\n",
    "    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def draw_styled_landmarks(image, results):\n",
    "    # Draw face connections\n",
    "    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, \n",
    "                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), \n",
    "                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)\n",
    "                             ) \n",
    "    # Draw pose connections\n",
    "    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,\n",
    "                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), \n",
    "                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)\n",
    "                             ) \n",
    "    # Draw left hand connections\n",
    "    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, \n",
    "                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), \n",
    "                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)\n",
    "                             ) \n",
    "    # Draw right hand connections  \n",
    "    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, \n",
    "                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), \n",
    "                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)\n",
    "                             ) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "cap = cv2.VideoCapture(0)\n",
    "# Set mediapipe model \n",
    "with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:\n",
    "    while cap.isOpened():\n",
    "\n",
    "        # Read feed\n",
    "        ret, frame = cap.read()\n",
    "\n",
    "        # Make detections\n",
    "        image, results = mediapipe_detection(frame, holistic)\n",
    "        print(results)\n",
    "        \n",
    "        # Draw landmarks\n",
    "        draw_styled_landmarks(image, results)\n",
    "\n",
    "        # Show to screen\n",
    "        cv2.imshow('OpenCV Feed', image)\n",
    "\n",
    "        # Break gracefully\n",
    "        if cv2.waitKey(10) & 0xFF == ord('q'):\n",
    "            break\n",
    "    cap.release()\n",
    "    cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "draw_landmarks(frame, results)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x1d8945c0cf8>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "Uy8n8BBWmvTjP86nAAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. Extract Keypoint Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "21"
      ]
     },
     "execution_count": 114,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(results.left_hand_landmarks.landmark)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "pose = []\n",
    "for res in results.pose_landmarks.landmark:\n",
    "    test = np.array([res.x, res.y, res.z, res.visibility])\n",
    "    pose.append(test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [],
   "source": [
    "pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)\n",
    "face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)\n",
    "lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)\n",
    "rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
       "       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
       "       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
       "       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])"
      ]
     },
     "execution_count": 109,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \n",
    "    if results.face_landmarks \n",
    "    else np.zeros(1404)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def extract_keypoints(results):\n",
    "    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)\n",
    "    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)\n",
    "    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)\n",
    "    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)\n",
    "    return np.concatenate([pose, face, lh, rh])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "metadata": {},
   "outputs": [],
   "source": [
    "result_test = extract_keypoints(results)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0.3835876 ,  0.47759178, -0.77978629, ...,  0.        ,\n",
       "        0.        ,  0.        ])"
      ]
     },
     "execution_count": 123,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [],
   "source": [
    "np.save('0', result_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0.3835876 ,  0.47759178, -0.77978629, ...,  0.        ,\n",
       "        0.        ,  0.        ])"
      ]
     },
     "execution_count": 125,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.load('0.npy')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4. Setup Folders for Collection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Path for exported data, numpy arrays\n",
    "DATA_PATH = os.path.join('MP_Data') \n",
    "\n",
    "# Actions that we try to detect\n",
    "actions = np.array(['hello', 'thanks', 'iloveyou'])\n",
    "\n",
    "# Thirty videos worth of data\n",
    "no_sequences = 30\n",
    "\n",
    "# Videos are going to be 30 frames in length\n",
    "sequence_length = 30\n",
    "\n",
    "# Folder start\n",
    "start_folder = 30"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "for action in actions: \n",
    "    dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))\n",
    "    for sequence in range(1,no_sequences+1):\n",
    "        try: \n",
    "            os.makedirs(os.path.join(DATA_PATH, action, str(dirmax+sequence)))\n",
    "        except:\n",
    "            pass"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 5. Collect Keypoint Values for Training and Testing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "cap = cv2.VideoCapture(0)\n",
    "# Set mediapipe model \n",
    "with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:\n",
    "    \n",
    "    # NEW LOOP\n",
    "    # Loop through actions\n",
    "    for action in actions:\n",
    "        # Loop through sequences aka videos\n",
    "        for sequence in range(start_folder, start_folder+no_sequences):\n",
    "            # Loop through video length aka sequence length\n",
    "            for frame_num in range(sequence_length):\n",
    "\n",
    "                # Read feed\n",
    "                ret, frame = cap.read()\n",
    "\n",
    "                # Make detections\n",
    "                image, results = mediapipe_detection(frame, holistic)\n",
    "\n",
    "                # Draw landmarks\n",
    "                draw_styled_landmarks(image, results)\n",
    "                \n",
    "                # NEW Apply wait logic\n",
    "                if frame_num == 0: \n",
    "                    cv2.putText(image, 'STARTING COLLECTION', (120,200), \n",
    "                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)\n",
    "                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), \n",
    "                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)\n",
    "                    # Show to screen\n",
    "                    cv2.imshow('OpenCV Feed', image)\n",
    "                    cv2.waitKey(500)\n",
    "                else: \n",
    "                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), \n",
    "                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)\n",
    "                    # Show to screen\n",
    "                    cv2.imshow('OpenCV Feed', image)\n",
    "                \n",
    "                # NEW Export keypoints\n",
    "                keypoints = extract_keypoints(results)\n",
    "                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))\n",
    "                np.save(npy_path, keypoints)\n",
    "\n",
    "                # Break gracefully\n",
    "                if cv2.waitKey(10) & 0xFF == ord('q'):\n",
    "                    break\n",
    "                    \n",
    "    cap.release()\n",
    "    cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "metadata": {},
   "outputs": [],
   "source": [
    "cap.release()\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 6. Preprocess Data and Create Labels and Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from tensorflow.keras.utils import to_categorical"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "label_map = {label:num for num, label in enumerate(actions)}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'hello': 0, 'thanks': 1, 'iloveyou': 2}"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "label_map"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "sequences, labels = [], []\n",
    "for action in actions:\n",
    "    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):\n",
    "        window = []\n",
    "        for frame_num in range(sequence_length):\n",
    "            res = np.load(os.path.join(DATA_PATH, action, str(sequence), \"{}.npy\".format(frame_num)))\n",
    "            window.append(res)\n",
    "        sequences.append(window)\n",
    "        labels.append(label_map[action])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(180, 30, 1662)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.array(sequences).shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(180,)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.array(labels).shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = np.array(sequences)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(180, 30, 1662)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = to_categorical(labels).astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(9, 3)"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 7. Build and Train LSTM Neural Network"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import LSTM, Dense\n",
    "from tensorflow.keras.callbacks import TensorBoard"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "log_dir = os.path.join('Logs')\n",
    "tb_callback = TensorBoard(log_dir=log_dir)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))\n",
    "model.add(LSTM(128, return_sequences=True, activation='relu'))\n",
    "model.add(LSTM(64, return_sequences=False, activation='relu'))\n",
    "model.add(Dense(64, activation='relu'))\n",
    "model.add(Dense(32, activation='relu'))\n",
    "model.add(Dense(actions.shape[0], activation='softmax'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {
    "scrolled": True
   },
   "outputs": [],
   "source": [
    "model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "lstm (LSTM)                  (None, 30, 64)            442112    \n",
      "_________________________________________________________________\n",
      "lstm_1 (LSTM)                (None, 30, 128)           98816     \n",
      "_________________________________________________________________\n",
      "lstm_2 (LSTM)                (None, 64)                49408     \n",
      "_________________________________________________________________\n",
      "dense (Dense)                (None, 64)                4160      \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 32)                2080      \n",
      "_________________________________________________________________\n",
      "dense_2 (Dense)              (None, 3)                 99        \n",
      "=================================================================\n",
      "Total params: 596,675\n",
      "Trainable params: 596,675\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 8. Make Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "res = model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'hello'"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "actions[np.argmax(res[4])]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'hello'"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "actions[np.argmax(y_test[4])]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 9. Save Weights"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('action.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 217,
   "metadata": {},
   "outputs": [],
   "source": [
    "del model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.load_weights('action.h5')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 10. Evaluation using Confusion Matrix and Accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import multilabel_confusion_matrix, accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "yhat = model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "ytrue = np.argmax(y_test, axis=1).tolist()\n",
    "yhat = np.argmax(yhat, axis=1).tolist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[5, 0],\n",
       "        [0, 4]],\n",
       "\n",
       "       [[5, 0],\n",
       "        [0, 4]],\n",
       "\n",
       "       [[8, 0],\n",
       "        [0, 1]]], dtype=int64)"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "multilabel_confusion_matrix(ytrue, yhat)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.0"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(ytrue, yhat)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 11. Test in Real Time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "from scipy import stats"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "colors = [(245,117,16), (117,245,16), (16,117,245)]\n",
    "def prob_viz(res, actions, input_frame, colors):\n",
    "    output_frame = input_frame.copy()\n",
    "    for num, prob in enumerate(res):\n",
    "        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)\n",
    "        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)\n",
    "        \n",
    "    return output_frame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 262,
   "metadata": {
    "collapsed": True
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x1daff6aa278>"
      ]
     },
     "execution_count": 262,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1296x1296 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(18,18))\n",
    "plt.imshow(prob_viz(res, actions, image, colors))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. New detection variables\n",
    "sequence = []\n",
    "sentence = []\n",
    "predictions = []\n",
    "threshold = 0.5\n",
    "\n",
    "cap = cv2.VideoCapture(0)\n",
    "# Set mediapipe model \n",
    "with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:\n",
    "    while cap.isOpened():\n",
    "\n",
    "        # Read feed\n",
    "        ret, frame = cap.read()\n",
    "\n",
    "        # Make detections\n",
    "        image, results = mediapipe_detection(frame, holistic)\n",
    "        print(results)\n",
    "        \n",
    "        # Draw landmarks\n",
    "        draw_styled_landmarks(image, results)\n",
    "        \n",
    "        # 2. Prediction logic\n",
    "        keypoints = extract_keypoints(results)\n",
    "        sequence.append(keypoints)\n",
    "        sequence = sequence[-30:]\n",
    "        \n",
    "        if len(sequence) == 30:\n",
    "            res = model.predict(np.expand_dims(sequence, axis=0))[0]\n",
    "            print(actions[np.argmax(res)])\n",
    "            predictions.append(np.argmax(res))\n",
    "            \n",
    "            \n",
    "        #3. Viz logic\n",
    "            if np.unique(predictions[-10:])[0]==np.argmax(res): \n",
    "                if res[np.argmax(res)] > threshold: \n",
    "                    \n",
    "                    if len(sentence) > 0: \n",
    "                        if actions[np.argmax(res)] != sentence[-1]:\n",
    "                            sentence.append(actions[np.argmax(res)])\n",
    "                    else:\n",
    "                        sentence.append(actions[np.argmax(res)])\n",
    "\n",
    "            if len(sentence) > 5: \n",
    "                sentence = sentence[-5:]\n",
    "\n",
    "            # Viz probabilities\n",
    "            image = prob_viz(res, actions, image, colors)\n",
    "            \n",
    "        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)\n",
    "        cv2.putText(image, ' '.join(sentence), (3,30), \n",
    "                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)\n",
    "        \n",
    "        # Show to screen\n",
    "        cv2.imshow('OpenCV Feed', image)\n",
    "\n",
    "        # Break gracefully\n",
    "        if cv2.waitKey(10) & 0xFF == ord('q'):\n",
    "            break\n",
    "    cap.release()\n",
    "    cv2.destroyAllWindows()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "action",
   "language": "python",
   "name": "action"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}