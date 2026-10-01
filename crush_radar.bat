@echo off
REM OpenJev crush radar - double-click launcher.
REM Starts the local hub and opens the page. Config lives in %USERPROFILE%\.openjev\llm.json
REM (set once with: python -m openjev.llm_config).
cd /d "%~dp0"
python -X utf8 -m openjev.crush_bot --port 8793 --open
pause