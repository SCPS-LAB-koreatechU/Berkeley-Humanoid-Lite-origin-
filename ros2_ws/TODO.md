# 해야할일 — 측정 및 손목 결합부

`berkeley_humanoid_lite_v1arm.urdf`에서 아직 **측정되지 않은 값**과 그 값을 채우는 절차.
모든 값은 `src/berkeley_humanoid_lite_description/config/arm_attachment.yaml`에 있고,
고친 뒤에는 `python3 generate_urdf.py`로 재생성한다 (`install/`은 symlink라 재빌드 불필요).

## A. 손목 결합부 (elbow_roll ↔ V1 전완 마운트)

현재 `mount.right = xyz [0,0,0], rpy [0,0,0]` — 어느 CAD에도 어댑터 부품이 없어서 넣은 추측값.
방향(+Z가 원위 방향)은 맞지만 **스탠드오프 거리와 롤 각**은 미확인.

- [ ] 실물 어댑터(또는 설계 예정 어댑터)의 치수 확보: elbow_roll 플랜지면 → V1 전완 기준면 거리, 볼트 패턴의 회전 오프셋
- [ ] 튜닝 모델로 슬라이더 맞추기
  ```bash
  cd ros2_ws && source /opt/ros/humble/setup.bash && source install/setup.bash
  ros2 launch berkeley_humanoid_lite_description display.launch.py model:=tuning
  ros2 run berkeley_humanoid_lite_description read_mount.py --watch
  ```
  `dexhand_mount_{x,y,z,yaw,pitch,roll}` 슬라이더를 움직여 손이 맞는 자리에 오면 출력 블록을 `mount.right`에 붙여넣기
- [ ] 왼팔은 `mount.left: mirror`(y·roll·yaw 부호 반전)로 유도됨 — 좌우 어댑터가 다르면 명시적 xyz/rpy로 교체
- [ ] `read_mount.py` 독스트링이 `config/dexhand_mount.yaml`을 가리키는데 실제 파일은 `arm_attachment.yaml` — 문구 수정
- [ ] 결합 후 손목 3축(`wrist_pitch_lower ±30°`, `wrist_yaw ±25°`, `wrist_pitch_upper ±20°`)이 전완 메시와 간섭 없는지 RViz/MoveIt 셀프콜리전으로 확인, 필요하면 SRDF 재생성(`generate_srdf.py`)

## B. 손가락 커플링 비율 (`hand.coupled`)

`Flexor ← Pitch = 1.0`, `DIP ← Flexor = 1.0`은 플레이스홀더. 지금 문서의 도달거리·핀치 가능 여부 수치가 전부 이 값에 의존.

- [ ] **커플링 존재 여부부터 확인**: 손을 고정하고 너클 하나만 전 구간 구동 → 중절·말절이 따라 도는지 눈으로 확인. 안 돌면 두 배율을 0으로 두고 끝
- [ ] 촬영 준비: 손가락 모듈을 클램프로 고정, 카메라를 손바닥 **y축 방향**(손가락 굴곡축과 평행)으로 고정. 핸드헬드 금지 — 포즈마다 원근 왜곡이 달라져 비율이 편향됨
- [ ] 9포즈 스윕을 영상으로 촬영 후 프레임 추출
  ```bash
  ffmpeg -i sweep.mp4 -vf fps=1 frame_%03d.png
  ```
- [ ] 관절 중심 추적 및 각도 추출 (`--finger`로 손가락 선택, 기본 Middle)
  ```bash
  python3 scripts/motion/track_finger_joints.py frames/ --finger Middle -o angles.csv
  ```
  카메라 조준 오류 경고가 나오면 재촬영 (사후 보정 불가)
- [ ] 비율 피팅 (관절 중심 클릭 모드가 가장 정밀, ±2 px → Flexor ±0.013, DIP ±0.022)
  ```bash
  python3 scripts/motion/fit_finger_coupling.py --from-points angles.csv
  ```
- [ ] 결과를 `hand.coupled.Flexor.multiplier`, `hand.coupled.DIP.multiplier`에 기입. 손가락마다 다르면 손가락별 값이 필요한지 판단
- [ ] 재생성 후 `python3 scripts/motion/analyze_hand.py`로 도달거리 표 갱신, `source/berkeley_humanoid_lite_motion/README.md`의 수치 업데이트

## C. 엄지 (고정 자세 + 어댑터)

- [ ] `digit_adapter.Thumb`: V2 엄지는 V1 마운트 대비 **15 mm** 어긋남 — 실제로 인쇄한 어댑터의 오프셋 측정 후 기입 (네 손가락은 2~3 mm 차이라 identity 유지 가능)
- [ ] 조립 시 엄지를 고정한 실제 각도를 `hand.thumb`에 기입. 기본값(joint zero)은 완전 신전·외전 자세라 핀치 불가 → `analyze_hand.py`가 제안하는 각도 참고

## D. 기타 미측정값 (우선순위 낮음)

- [ ] 전완 질량: 상류 관성은 강철 밀도 기준을 PLA로 스케일한 값(1240 kg/m³) — 완성 조립체 실측 무게로 교체
- [ ] 손끝 프레임 오프셋 `tip_frames.offset`: 메시에서 읽은 값, 실제 패드 접촉점과 비교 확인
- [ ] 서보 가동범위 `hand.actuated` (Pitch 0~0.95, Yaw ±0.30): 8서보 빌드의 실제 리밋과 대조

## 완료

- [x] RViz에서 DexHand가 빨갛게 보이던 문제 — `graft_v2_digits()`가 `silver` 재질 정의를 복사하지 않던 것을 수정 (`generate_urdf.py`)
