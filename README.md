# CopyCat 🐱

**CopyCat** faz a pipeline completa de imitação de movimento humano para robôs:

```
Vídeo → SMPL (GENMO) → Movimento de robô (GMR)
```

---

## Pré-requisitos

Certifique-se de que as duas ferramentas estejam instaladas e funcionando individualmente:

- **GENMO** — em `CopyCat/GENMO/`
- **GMR** — em `CopyCat/GMR/`

---

## Uso

Todos os comandos abaixo são executados a partir da pasta `CopyCat/`.

### Exemplo mínimo

```bash
python run_pipeline.py \
    --video /caminho/para/video.mp4 \
    --robot booster_t1
```

### Exemplo completo

```bash
python run_pipeline.py \
    --video /caminho/para/video.mp4 \
    --video_name minha_danca \
    --robot booster_t1 \
    --ckpt_path GENMO/inputs/checkpoints/s050000.ckpt \
    --orig_fps 30 \
    --save_path outputs/minha_danca/robot_motion.pkl \
    --record_video \
    --rate_limit
```

---

## Flags

### GENMO — vídeo → SMPL

| Flag | Padrão | Descrição |
|---|---|---|
| `--video` | *obrigatório* | Caminho para o vídeo de entrada **ou** uma pasta contendo múltiplos `.mp4` |
| `--video_name` | stem do arquivo | Nome usado na pasta de saída (ignorado se `--video` for uma pasta) |
| `--output_dir` | `GENMO/outputs/demo/<video_name>` | Pasta de saída do GENMO |
| `--ckpt_path` | `GENMO/inputs/checkpoints/s050000.ckpt` | Checkpoint do modelo |
| `--exp` | `genmo_lg` | Configuração de experimento |
| `--orig_fps` | `30` | FPS original do vídeo de entrada |
| `--static_cam` | ativo por padrão | Assume câmera estática (sem SLAM) |
| `--no_static_cam` | — | Ativa SLAM para câmera em movimento |

### GMR — SMPL → robô

| Flag | Padrão | Descrição |
|---|---|---|
| `--robot` | `unitree_g1` | Robô-alvo para o retargeting (ver lista abaixo) |
| `--save_path` | `<output_dir>/robot_motion_<robot>.pkl` | Onde salvar o movimento do robô |
| `--record_video` | desligado | Grava um vídeo da visualização |
| `--rate_limit` | desligado | Limita a taxa de reprodução ao FPS do humano |
| `--loop` | desligado | Repete o movimento no viewer indefinidamente |

### Robôs suportados

```
unitree_g1        unitree_g1_with_hands    unitree_h1       unitree_h1_2
booster_t1        booster_t1_29dof         stanford_toddy   fourier_n1
engineai_pm01     kuavo_s45                hightorque_hi    galaxea_r1pro
berkeley_humanoid_lite   booster_k1        pnd_adam_lite    openloong
tienkung
```

---

## Saídas

| Arquivo | Descrição |
|---|---|
| `<output_dir>/hmr4d_results.pt` | Estimativa SMPL do GENMO |
| `<output_dir>/robot_motion_<robot>.pkl` | Movimento retargetado do robô |
| `GMR/videos/<robot>_hmr4d_results.mp4` | Vídeo de visualização (se `--record_video`) |

---

## Estrutura

```
CopyCat/
├── run_pipeline.py   ← script principal
├── GENMO/            ← repositório GENMO
└── GMR/              ← repositório GMR
```
