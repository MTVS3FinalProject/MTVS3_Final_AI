import { useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom'; // 페이지 이동을 위한 hook
import * as faceMesh from '@mediapipe/face_mesh'; // MediaPipe Face Mesh
import * as cameraUtils from '@mediapipe/camera_utils';

function CapturePhoto() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null); // 사진을 찍기 위한 canvas
  const streamRef = useRef(null); // 스트림 저장
  const navigate = useNavigate(); // 페이지 이동을 위한 hook
  const blinkCountRef = useRef(0); // 깜빡임 카운트
  const lastBlinkTimeRef = useRef(0); // 마지막 깜빡임 시간
  const eyeClosedRef = useRef(false); // 눈이 닫힌 상태인지 추적

  // 기존 스트림 종료 함수
  const stopStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    }
  };

  // 카메라 시작 함수 (전면 카메라만 사용)
  const startCamera = useCallback(async () => {
    stopStream(); // 새 스트림 시작 전에 기존 스트림 종료
    try {
      const constraints = {
        video: {
          facingMode: 'user', // 전면 카메라 고정
        },
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      streamRef.current = stream; // 새 스트림 저장
    } catch (error) {
      console.error('Error accessing the camera', error);
      alert('카메라에 접근하는 데 문제가 발생했습니다. 권한을 확인하세요.');
    }
  }, []);

  // MediaPipe Face Mesh 설정 및 시작
  const startFaceMesh = useCallback(() => {
    const faceMeshInstance = new faceMesh.FaceMesh({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
    });

    faceMeshInstance.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    faceMeshInstance.onResults(onResults);

    if (videoRef.current) {
      const camera = new cameraUtils.Camera(videoRef.current, {
        onFrame: async () => {
          await faceMeshInstance.send({ image: videoRef.current });
        },
        width: 640,
        height: 480,
      });
      camera.start();
    }
  }, []);

  // 눈 깜빡임을 감지한 결과 처리 함수
  const onResults = (results) => {
    if (results.multiFaceLandmarks) {
      const leftEyeUpper = 159;
      const leftEyeLower = 145;
      const rightEyeUpper = 386;
      const rightEyeLower = 374;

      for (const landmarks of results.multiFaceLandmarks) {
        const leftEyeOpen = isEyeOpen(landmarks, leftEyeUpper, leftEyeLower);
        const rightEyeOpen = isEyeOpen(landmarks, rightEyeUpper, rightEyeLower);

        if (!leftEyeOpen && !rightEyeOpen && !eyeClosedRef.current) {
          eyeClosedRef.current = true;
          const currentTime = new Date().getTime();
          if (currentTime - lastBlinkTimeRef.current > 500) { // 0.5초 이후에만 카운트
            blinkCountRef.current++;
            lastBlinkTimeRef.current = currentTime;

            if (blinkCountRef.current === 2) {
              // 두 번 깜빡임이 감지되면 사진 촬영
              setTimeout(capturePhoto, 1000); // 1초 지연 후 사진 촬영
              blinkCountRef.current = 0; // 카운트 리셋
            }
          }
        }

        if (leftEyeOpen || rightEyeOpen) {
          eyeClosedRef.current = false;
        }
      }
    }
  };

  // 눈이 열려있는지 확인하는 함수
  const isEyeOpen = (landmarks, upperIndex, lowerIndex) => {
    const upper = landmarks[upperIndex];
    const lower = landmarks[lowerIndex];
    const eyeHeight = Math.abs(upper.y - lower.y);
    return eyeHeight > 0.015; // 눈이 열려있는지 확인하는 임계값
  };

  // 캡처 함수 (좌우 반전 포함)
  const capturePhoto = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video && canvas) {
      const context = canvas.getContext('2d');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // 좌우 반전 적용
      context.translate(canvas.width, 0);
      context.scale(-1, 1);

      // 비디오 프레임을 캔버스에 그리기 (좌우 반전된 상태로)
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      // 캡처한 이미지를 base64 URL로 변환
      const imageDataUrl = canvas.toDataURL('image/png');

      // 이미지를 새로운 페이지로 전달하고 이동
      navigate('/photo', { state: { photo: imageDataUrl } });
    }
  };

  useEffect(() => {
    startCamera(); // 컴포넌트가 마운트될 때 카메라 시작
    startFaceMesh(); // MediaPipe Face Mesh 시작
    return () => {
      stopStream(); // 컴포넌트가 언마운트될 때 스트림 종료
    };
  }, [startCamera, startFaceMesh]);

  return (
    <div>
      <h1>Take a Photo by Blinking Twice</h1>

      {/* 비디오 스트림 */}
      <video ref={videoRef} autoPlay playsInline style={{ width: '100%', maxHeight: '300px' }} />

      {/* Canvas 요소 - 캡처용 (화면에 보이지 않음) */}
      <canvas ref={canvasRef} style={{ display: 'none' }}></canvas>
    </div>
  );
}

export default CapturePhoto;
