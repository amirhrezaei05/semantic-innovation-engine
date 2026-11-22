@echo off
setlocal
if not exist .venv (
  py -3.11 -m venv .venv
  if errorlevel 1 goto :error
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto :error
streamlit run app.py
goto :eof
:error
echo Setup failed. Check that Python 3.11 is installed and review the error above.
pause
