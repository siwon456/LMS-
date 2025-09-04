import os
# 먼저, 원래 PATH 업데이트 코드를 추가합니다.
os.environ["PATH"] = r"C:\Users\User\anaconda3\pkgs\libflac-1.5.0-he0c23c2_0\Library\bin" + os.pathsep + os.environ["PATH"]

# 이후 monkey-patching: speech_recognition이 FLAC 변환기로 아래 경로를 직접 사용하도록 강제합니다.
import speech_recognition as sr
sr.audio.get_flac_converter = lambda: r"C:\Users\User\anaconda3\pkgs\libflac-1.5.0-he0c23c2_0\Library\bin\flac.exe"

# 진단: 확인용 코드 (선택 사항)
import shutil
print("FLAC found at:", shutil.which("flac"))
