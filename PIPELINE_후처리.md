# 후처리 → FBX 변환 파이프라인

GVHMR(전신) + HaMeR(손) 로 추정된 SMPL-X 모션(`.pt`)을 **후처리**(스무딩·트위스트 안정화)한 뒤 **Unreal용 FBX**로 변환하는 파이프라인 문서.

> ⚠️ **FBX 변환 단계(`export_to_unreal_fbx.py` 및 Blender 익스포트)는 검증 완료되어 동결되었습니다.** 손대지 마세요. 후처리 단계(`animate_postprocess.py`, `rotation_postprocess.py`)는 수정 가능.

---

## 1. 전체 흐름

```
 [입력] smplx_hmp_injected_full.pt        (GVHMR body + HaMeR hands, SMPL-X 파라미터)
    │
    │  ── Stage A: 후처리 (Python / venv) ─────────────────────────────
    │   animate_postprocess.py
    │     ├─ 회전 후처리  rotation_postprocess.py
    │     │     · body_pose : 연속성 → SG 스무딩 → 스윙-트위스트 ROM(팔꿈치) → 각속도 클램핑
    │     │     · 기타 회전 : 연속성 → SG 스무딩 → 각속도 클램핑
    │     │     · transl    : One-Euro
    │     ├─ SMPL-X FK (시각화용)              model/SMPLX_NEUTRAL.npz
    │     ├─ 스윙 단계 분류  swing_classifier.py (시각화/라켓용)
    │     └─ (옵션) PyVista Before/After 비교
    ▼
 [중간] Result/<stem>_postprocessed_<naming>.npz   (후처리된 SMPL-X 파라미터)
    │
    │  ── Stage B: FBX 변환 (Blender 5.1) ──────────────────────────────
    │   export_to_unreal_fbx.py   (-b smplx_template.blend 위에서 실행)
    │     ├─ convert_to_amass : npz → AMASS poses(165), Y-up→Z-up(root), betas
    │     ├─ smplx_add_animation(AMASS, hand_reference="RELAXED")  ← SMPL-X Blender 애드온
    │     │     · 애드온이 포즈를 본에 직접 대입 + 손은 hands_mean 가산
    │     ├─ bake_to_keyframes
    │     └─ export_scene.fbx  (Unreal 축: forward=X, up=Z)
    ▼
 [출력] Result/<base>_smplx_unreal.fbx        (메시 + 스켈레톤 + 애니메이션)
        Result/<base>_smplx_unreal_anim.fbx   (애니메이션 전용)
    │
    └─ run.bat 가 front_4/V2.fbx, front_v2.fbx 로 복사
```

---

## 2. 파일 구성

### 실행 파이프라인 (런타임)

| 파일 | 역할 |
|---|---|
| [run.bat](run.bat) | **오케스트레이션**: 후처리 → Blender 익스포트 → 결과 복사. ([fix_run_bat.py](fix_run_bat.py)가 생성) |
| [animate_postprocess.py](animate_postprocess.py) | **Stage A 진입점**. `.pt` 로드 → 회전 후처리 → FK → 스윙 분류 → `.npz` 저장 → (옵션)시각화 |
| [rotation_postprocess.py](rotation_postprocess.py) | **회전 후처리 핵심 모듈**. 연속성·Savitzky-Golay·스윙-트위스트 ROM·각속도 클램핑. (기존 One-Euro+XPBD 대체) |
| [swing_classifier.py](swing_classifier.py) | 스윙 단계/스트로크 분류 (시각화·라켓 방향용) |
| [export_to_unreal_fbx.py](export_to_unreal_fbx.py) | **Stage B**. npz → AMASS → 애드온 애니메이션 → bake → FBX 익스포트 **(동결)** |
| [model/SMPLX_NEUTRAL.npz](model/SMPLX_NEUTRAL.npz) | SMPL-X 모델 (Stage A의 FK용) |
| smplx_template.blend | Blender 템플릿 씬 (SMPL-X 애드온 컨텍스트, Stage B의 베이스) |

### 외부 의존성

| 항목 | 비고 |
|---|---|
| Python venv | `torch, numpy, scipy, smplx, pyvista` |
| Blender 5.1 | `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe` |
| SMPL-X Blender 애드온 | `…/Blender/5.1/extensions/user_default/smplx_blender_addon` — `smplx_add_animation` 제공 |

### 검증 · 참고 (런타임 아님)

| 경로 | 역할 |
|---|---|
| [postprocess_lab/](postprocess_lab/) | 후처리 기법 비교 실험실: [metrics.py](postprocess_lab/metrics.py)(지표), [methods.py](postprocess_lab/methods.py)(5가지 기법), [baseline.py](postprocess_lab/baseline.py), [compare.py](postprocess_lab/compare.py) |
| [cpp_reference/RotationPostprocess.h](cpp_reference/RotationPostprocess.h) | 회전 후처리 파이프라인의 C++ 이식 참고본 (UE 등) |
| [xpbd_constraints.py](xpbd_constraints.py) | **레거시**. 구 XPBD 관절 제한 — 현재 미사용(롤백/비교용 보존) |
| Result/verify_*.py, diag_*.py | FBX/손가락/평균 검증·진단 스크립트 |

---

## 3. 단계별 동작

### Stage A — 후처리 ([animate_postprocess.py](animate_postprocess.py))

1. **로드**: `torch.load(.pt)['smpl_params_global']` — `body_pose (T,21,3)`, `global_orient`, `left/right_hand_pose (T,45)`, `transl`, `betas`, `jaw/leye/reye_pose`.
2. **회전 후처리 (Step 1)** — 키별로 처리:
   - `body_pose` → `stabilize_body_pose()` : 관절별 **(1)연속성 → (2)SG 스무딩 → (3)스윙-트위스트 ROM → (4)각속도 클램핑**. 트위스트 ROM은 [BODY_TWIST_LIMITS_DEG](rotation_postprocess.py#L27)에 등록된 **팔꿈치(±90°)에만** 적용(손목/어깨는 제외 — 그립 보존).
   - 그 외 회전키(`global_orient`, 손, 턱/눈) → `smooth_rotations()` : (1)연속성 → (2)SG 스무딩 → (4)각속도 클램핑.
   - `transl` → One-Euro 필터.
3. **FK**: SMPL-X 모델로 raw/후처리 메시·관절 계산 (시각화 데이터).
4. **스윙 분류 (Step 3)**: `classify_swing_phases()` (PyVista 라켓 표시용).
5. **저장**: `Result/<stem>_postprocessed_<naming>.npz` (global_orient, body_pose, 손, transl, betas, gender). **npz 저장은 시각화보다 먼저** 일어나므로 `--visualize` 창을 닫지 않아도 다음 단계가 읽을 파일은 이미 생성됨.
6. **(옵션) 시각화**: `--visualize` 시 Before/After 메시 + 라켓 비교 창.

### Stage B — FBX 변환 ([export_to_unreal_fbx.py](export_to_unreal_fbx.py), 동결)

1. **convert_to_amass**: 후처리 npz → AMASS 포맷 `poses(T,165)`. `global_orient`과 `trans`만 `_R_Y2Z`로 Y-up→Z-up 변환(손/몸 로컬 회전은 변환 안 함).
2. **smplx_add_animation(AMASS, hand_reference="RELAXED")**: 애드온이 새 SMPL-X 아마추어+메시를 만들고 포즈를 본에 **직접 대입**. `RELAXED`로 손에 모델 `hands_mean`을 더해(=애드온 relaxed) 손가락 그립을 정상화. root 본에만 -90°(Y-up→Z-up) 보정.
3. **bake_to_keyframes**: 포즈를 키프레임으로 베이크.
4. **export_scene.fbx**: Unreal 축(forward=X, up=Z, primary_bone_axis=Y)으로 익스포트.
   - 출력: `<base>_smplx_unreal.fbx`(전체), `<base>_smplx_unreal_anim.fbx`(애니메이션 전용).

---

## 4. 실행 방법

```bat
:: 전체 파이프라인 (입력 .pt 와 복사 경로는 run.bat 안에 하드코딩)
run.bat
```

개별 실행:

```bat
:: Stage A — 후처리 (--visualize 빼면 창 없이 npz만 저장)
venv\Scripts\python animate_postprocess.py "smplx_hmp_injected_full.pt" --naming smplx

:: Stage B — FBX 변환
"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" -b smplx_template.blend ^
    -P export_to_unreal_fbx.py -- ^
    "Result\smplx_hmp_injected_full_postprocessed_smplx.npz" "Result"
```

> 입력 `.pt`를 바꾸려면 [fix_run_bat.py](fix_run_bat.py)의 경로를 수정 후 `python fix_run_bat.py`로 `run.bat`을 재생성.

---

## 5. 후처리 파라미터 튜닝 ([rotation_postprocess.py](rotation_postprocess.py))

| 파라미터 | 기본값 | 의미 / 조정 |
|---|---|---|
| `window` (SG 윈도우) | 9 (≈0.3s) | 노이즈 심하면 11~13, 지연 줄이려면 7 |
| `max_deg_per_frame` (각속도 상한) | 30 | 더 부드럽게=15, 빠른 모션 보존=40 |
| `BODY_TWIST_LIMITS_DEG` | 팔꿈치 ±90° | 서브 회내 살리려면 ±120°. **손목/어깨는 의도적으로 제외**(PCA 트위스트축이 굽힘과 섞여 그립을 꺾음) |

### 설계 메모 (왜 이렇게?)
- **One-Euro+XPBD(Euler) 제거 이유**: Euler-XYZ 분해는 짐벌락/언랩 불안정으로 손목 트위스트(Z축)에서 플래싱을 유발. 실측에서 XPBD가 플래싱을 오히려 악화(손목 각속도 33→90°/f)시켰음.
- **승자 파이프라인**: 연속성+Savitzky-Golay+스윙-트위스트(unwrap)+각속도 클램핑 조합이 트위스트(178→90°)·플래싱(jerk 8.8→1.5)·불연속(flip 4→0)을 동시 해결. 근거/재현은 [postprocess_lab/compare.py](postprocess_lab/compare.py).
- **손목 트위스트 ROM 제외 이유**: 손목 트위스트는 라켓 그립의 실제 방향이라 클램핑하면 손이 꺾임. 손목은 스무딩만으로 원본을 ~5°로 보존.
- **FBX 손가락 `RELAXED` 이유**: SMPL-X 손 포즈는 flat 기준 상대값 → 본 적용 시 `hands_mean`을 더해야 정상. FLAT은 평균 누락으로 손가락이 ~49° 틀어짐.

---

## 6. 검증 도구

```bat
:: 후처리 기법 5종 비교 (지표 테이블)
venv\Scripts\python postprocess_lab\compare.py [data.pt]

:: 변환 전 npz ↔ 변환 후 FBX 손가락 값 일치 검증
"…\blender.exe" -b -P Result\verify_npz_vs_fbx.py -- <npz> <fbx>
```

지표: 측지 각속도(플래싱), jerk(떨림), 스윙-트위스트각(과도 트위스트), 쿼터니언 부호 뒤집힘, 원본 대비 편차(충실도). 정의는 [postprocess_lab/metrics.py](postprocess_lab/metrics.py).
