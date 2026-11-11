# 로그인 없이 받을 수 있는 공개 원천을 data/raw/에 내려받는다. 이미 있는 파일은 건너뜀.
#  - 행정동 경계 2020-07 (admdongkor, CC BY 4.0)
#  - 서울 생활인구(행정동) 2017–2022 반기 zip 12개 (OA-14991, 약 3.5GB)
# 인허가 2종·상가정보는 브라우저로 받아야 함(README 참고).
import os,subprocess,zipfile
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R=f'{O}/data/raw'
def get(path,args):
    if os.path.exists(path) and os.path.getsize(path)>0: print('있음',os.path.relpath(path,O)); return
    os.makedirs(os.path.dirname(path),exist_ok=True); subprocess.run(['curl','-sL','--retry','3','-o',path,*args],check=True); print('받음',os.path.relpath(path,O))
get(f'{R}/행정동경계/HangJeongDong_ver20200701.geojson',['https://raw.githubusercontent.com/vuski/admdongkor/master/ver20200701/HangJeongDong_ver20200701.geojson'])
seq=2213   # 열린데이터광장 파일 번호: 2017 상반기=2213부터 반기마다 1씩
for y in range(2017,2023):
    for h in ('상반기','하반기'):
        p=f'{R}/생활인구/LOCAL_PEOPLE_DONG_{y}_{h}.zip'
        get(p,['-X','POST','https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false','-d',f'infId=OA-14991&seq={seq}&infSeq=3&seqNo='])
        zipfile.ZipFile(p).testzip(); seq+=1
