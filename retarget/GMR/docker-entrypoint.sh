#!/bin/bash
# Inicia Xvfb em background para fornecer display virtual ao GLFW.
# mjv.launch_passive() precisa de X11 independente de MUJOCO_GL.
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX &
export DISPLAY=:99
exec "$@"
