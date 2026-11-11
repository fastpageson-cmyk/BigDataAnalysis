# web/index.src.html → web/index.html: Material 3 CSS(google-design 스킬의 tokens·component-tokens·google.css)와 Leaflet CSS를 인라인한다.
#  이유: 아티팩트는 외부 스타일시트를 Google Fonts에서만 불러올 수 있다. Leaflet CSS는 web/leaflet-1.9.4.css에 한 번 받아 둔다.
import os,urllib.request
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); W=f'{O}/web'
SK=os.path.expanduser('~/.claude/skills/google-design')
lf=f'{W}/leaflet-1.9.4.css'
if not os.path.exists(lf): urllib.request.urlretrieve('https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css',lf)
m3='\n'.join(open(f'{SK}/{n}',encoding='utf-8').read() for n in ['tokens.css','component-tokens.css','google.css'])
s=open(f'{W}/index.src.html',encoding='utf-8').read().replace('/*M3_CSS*/',m3).replace('/*LEAFLET_CSS*/',open(lf,encoding='utf-8').read())
open(f'{W}/index.html','w',encoding='utf-8').write(s); print('web/index.html',round(len(s)/1e3),'KB')
