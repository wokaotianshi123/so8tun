import os
import json
import wave
import struct
import math
import warnings
from io import BytesIO
from urllib.parse import parse_qs
import multipart  # python-multipart

# 忽略无关警告
warnings.filterwarnings("ignore")

# 强制指定ffmpeg路径
os.environ["FFMPEG_PATH"] = "/usr/bin/ffmpeg"
os.environ["PYDUB_FFMPEG_PATH"] = "/usr/bin/ffmpeg"
from pydub import AudioSegment
import numpy as np

# 音频特征提取（复用核心逻辑）
def extract_audio_features(audio_path):
    try:
        if not audio_path.lower().endswith((".mp3", ".wav")):
            return {"success": False, "error": "仅支持MP3/WAV格式"}
        
        AudioSegment.converter = "/usr/bin/ffmpeg"
        temp_wav_path = "/tmp/temp_audio.wav"
        
        if audio_path.lower().endswith(".mp3"):
            audio = AudioSegment.from_mp3(audio_path)
        else:
            audio = AudioSegment.from_wav(audio_path)
        
        audio = audio.set_channels(1).set_sample_width(2).set_frame_rate(44100)
        audio.export(temp_wav_path, format="wav")

        with wave.open(temp_wav_path, 'rb') as wf:
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            if n_frames == 0:
                os.remove(temp_wav_path)
                return {"success": False, "error": "音频文件为空"}
            duration = n_frames / framerate

            frames = wf.readframes(n_frames)
            fmt = f"{n_frames}h"
            data = struct.unpack(fmt, frames)
            data = np.array(data, dtype=np.float32) / 32768.0

        os.remove(temp_wav_path)

        noise_frames = min(int(10 * framerate), len(data))
        noise_data = data[:noise_frames]
        noise_energy = np.sum(np.abs(noise_data)) / len(noise_data) if len(noise_data) > 0 else 0

        voice_start = max(int(10 * framerate), 0)
        voice_end = min(int(60 * framerate), len(data))
        voice_data = data[voice_start:voice_end] if voice_end > voice_start else data
        if len(voice_data) == 0:
            return {"success": False, "error": "人声段无有效音频数据"}

        voice_rms = np.sqrt(np.mean(np.square(voice_data))) if len(voice_data) > 0 else 0
        voice_avg_volume = 20 * math.log10(voice_rms + 1e-8)
        voice_peak_volume = 20 * math.log10(np.max(np.abs(voice_data)) + 1e-8) if len(voice_data) > 0 else -80
        voice_min_volume = 20 * math.log10(np.min(np.abs(voice_data[voice_data > 0])) + 1e-8) if len(voice_data[voice_data > 0]) > 0 else -80
        voice_dynamic_range = voice_peak_volume - voice_min_volume
        high_freq_energy = np.mean(np.abs(np.diff(voice_data))) if len(voice_data) > 1 else 0

        return {
            "duration": duration,
            "noise_energy": noise_energy,
            "voice_avg_volume": voice_avg_volume,
            "voice_peak_volume": voice_peak_volume,
            "voice_dynamic_range": voice_dynamic_range,
            "high_freq_energy": high_freq_energy,
            "success": True
        }
    except Exception as e:
        return {"success": False, "error": f"特征提取失败：{str(e)}"}

# 参数生成函数（无修改）
def generate_so8_params(form_data, audio_features=None):
    params = {
        "basic_knobs": "",
        "noise_reduction": "-70db",
        "eq_params": "",
        "compressor_params": "",
        "reverb_params": "",
        "optimize_tips": ""
    }
    
    mic_volume = 24
    record_volume = 25
    if "动圈" in form_data.get("mic_model", ""):
        mic_volume = 26
    elif "电容" in form_data.get("mic_model", ""):
        mic_volume = 22
    if form_data.get("usage") == "sing-live":
        record_volume = 24
    if audio_features and audio_features.get("success"):
        if audio_features["voice_peak_volume"] > -10:
            record_volume = max(20, record_volume - 3)
        elif audio_features["voice_peak_volume"] < -25:
            record_volume = min(28, record_volume + 2)
    params["basic_knobs"] = f"话筒音量：{mic_volume} | 耳机音量：25 | 伴奏音量：23 | 录音音量：{record_volume} | 音效音量：18"
    
    if audio_features and audio_features.get("success"):
        if audio_features["noise_energy"] > 0.1:
            params["noise_reduction"] = "-60db"
        elif audio_features["noise_energy"] < 0.02:
            params["noise_reduction"] = "-82db"
    else:
        noise_level = form_data.get("noise_level")
        if noise_level == "quiet":
            params["noise_reduction"] = "-80db"
        elif noise_level == "slight":
            params["noise_reduction"] = "-75db"
        elif noise_level == "loud":
            params["noise_reduction"] = "-65db"
    if "小声唱歌易断音" in form_data.get("voice_problem", ""):
        params["noise_reduction"] = f"{int(params['noise_reduction'].replace('db', '')) - 8}db"
    
    eq_base = "q值统一4.5 | 高通12档(40hz) | 低通12档(12000hz) | "
    eq_custom = ""
    if audio_features and audio_features.get("success"):
        if audio_features["high_freq_energy"] > 0.1:
            eq_custom += "5350hz增益-2.5 | 9500hz增益-2 | "
        elif audio_features["high_freq_energy"] < 0.02:
            eq_custom += "5350hz增益+3 | 9500hz增益+2.5 | "
    if "有鼻音" in form_data.get("voice_problem", ""):
        eq_custom += "1350hz增益-2.5 | "
    if form_data.get("voice_type") == "thin":
        eq_custom += "100hz增益+3.5 | 450hz增益+2.5 | "
    if form_data.get("voice_type") == "thick":
        eq_custom += "100hz增益-1.5 | 280hz增益-1.5 | "
    params["eq_params"] = eq_base + eq_custom + "100hz增益+2 | 140hz增益-1 | 280hz增益0 | 450hz增益+1.5 | 850hz增益-0.5 | 2800hz增益+1 | 10000hz增益+1.5"
    
    compressor_base = "压缩比4:1 | "
    threshold = "-15db（小嗓门）"
    attack_time = 440
    release_time = 1700
    gain_compensation = 2
    if audio_features and audio_features.get("success"):
        if audio_features["voice_avg_volume"] > -15:
            threshold = "-25db（大嗓门）"
        elif audio_features["voice_avg_volume"] < -25:
            threshold = "-12db（小嗓门）"
        if audio_features["voice_dynamic_range"] > 20:
            release_time = 1800
        gain_compensation = 3 if audio_features["voice_avg_volume"] < -20 else 2
    if "快歌" in form_data.get("song_style", "") or "rap" in form_data.get("song_style", ""):
        attack_time = 800
        release_time = 1500
    params["compressor_params"] = f"{compressor_base}阈值：{threshold} | 开始时间：{attack_time} | 释放时间：{release_time} | 增益补偿：+{gain_compensation}"
    
    dry_wet = 82
    reverb_vol = 22
    if audio_features and audio_features.get("success") and audio_features["voice_avg_volume"] < -20:
        dry_wet = 78
        reverb_vol = 18
    usage = form_data.get("usage")
    if usage == "sing-live":
        params["reverb_params"] = f"干湿比{dry_wet}% | 混响音量{reverb_vol} | 预延迟18ms | 混响时间4600ms"
    elif usage == "chat":
        params["reverb_params"] = f"干湿比{dry_wet - 10}% | 混响音量{reverb_vol - 7} | 预延迟15ms | 混响时间3000ms"
    elif usage == "audio-book":
        params["reverb_params"] = f"干湿比{dry_wet - 20}% | 混响音量{reverb_vol - 12} | 预延迟10ms | 混响时间2000ms"
    else:
        params["reverb_params"] = f"干湿比{dry_wet - 2}% | 混响音量{reverb_vol - 2} | 预延迟20ms | 混响时间4000ms"
    
    tips = []
    if audio_features and audio_features.get("success"):
        if audio_features["voice_peak_volume"] > -5:
            tips.append("1. 人声峰值音量较高，建议录音音量再降低2-3，避免爆音；")
        elif audio_features["voice_avg_volume"] < -30:
            tips.append("1. 人声平均音量较低，建议话筒音量提高2-3，压缩增益补偿+4；")
        if audio_features["noise_energy"] > 0.15:
            tips.append("2. 噪音能量较高，建议增加吸音棉/关闭门窗，降噪阈值可微调至-55db；")
        if audio_features["high_freq_energy"] > 0.1:
            tips.append("3. 人声高频过强，若仍刺耳可将5350hz增益再减1；")
        elif audio_features["high_freq_energy"] < 0.02:
            tips.append("3. 人声高频不足，若声音发闷可将9500hz增益再+1；")
    if not tips:
        tips.append("1. 若唱歌爆音，可将录音音量再降低1-2；2. 若混响盖过人声，可降低混响音量或干湿比；")
    if "鼻音" in form_data.get("voice_problem", ""):
        tips.append("4. 若鼻音仍明显，可将1350hz增益再减1；")
    params["optimize_tips"] = "".join(tips)
    
    return params

# 解析multipart/form-data请求
def parse_multipart(environ):
    form_data = {}
    audio_file = None
    audio_filename = None
    
    try:
        parser = multipart.FormParser(environ, strict=True)
        for part in parser:
            if part.filename is None:
                # 文本字段
                form_data[part.name] = part.value.decode('utf-8')
            else:
                # 音频文件
                audio_filename = part.filename
                audio_data = part.value
                # 保存临时文件
                temp_path = "/tmp/upload_audio"
                with open(temp_path, 'wb') as f:
                    f.write(audio_data)
                audio_file = temp_path
    except Exception as e:
        return form_data, None, None, str(e)
    
    return form_data, audio_file, audio_filename, None

# 核心WSGI处理函数
def application(environ, start_response):
    # 跨域头（全局）
    headers = [
        ('Access-Control-Allow-Origin', '*'),
        ('Access-Control-Allow-Methods', 'POST, OPTIONS'),
        ('Access-Control-Allow-Headers', 'Content-Type, Multipart/form-data'),
        ('Content-Type', 'application/json; charset=utf-8')
    ]
    
    # 处理OPTIONS预检请求
    if environ['REQUEST_METHOD'] == 'OPTIONS':
        start_response('200 OK', headers)
        return [b'{}']
    
    # 处理POST请求
    if environ['REQUEST_METHOD'] == 'POST':
        try:
            # 解析表单数据
            form_data, audio_file, audio_filename, parse_err = parse_multipart(environ)
            if parse_err:
                response = json.dumps({
                    "code": 400,
                    "msg": f"表单解析失败：{parse_err}",
                    "audio_features": None,
                    "params": {}
                }).encode('utf-8')
                start_response('400 Bad Request', headers)
                return [response]
            
            # 处理音频文件
            audio_features = None
            if audio_file:
                # 检查文件大小
                file_size = os.path.getsize(audio_file)
                if file_size > 5 * 1024 * 1024:
                    os.remove(audio_file)
                    response = json.dumps({
                        "code": 400,
                        "msg": "音频文件大小不能超过5MB",
                        "audio_features": None,
                        "params": {}
                    }).encode('utf-8')
                    start_response('400 Bad Request', headers)
                    return [response]
                
                # 提取特征
                audio_features = extract_audio_features(audio_file)
                os.remove(audio_file)
            
            # 生成参数
            params = generate_so8_params(form_data, audio_features)
            
            # 构建响应
            response = json.dumps({
                "code": 200,
                "msg": "参数生成成功",
                "audio_features": audio_features,
                "params": params
            }, ensure_ascii=False).encode('utf-8')
            
            start_response('200 OK', headers)
            return [response]
        
        except Exception as e:
            response = json.dumps({
                "code": 500,
                "msg": f"服务器错误：{str(e)}",
                "audio_features": None,
                "params": {}
            }, ensure_ascii=False).encode('utf-8')
            start_response('500 Internal Server Error', headers)
            return [response]
    
    # 其他请求方法
    response = json.dumps({
        "code": 405,
        "msg": "仅支持POST/OPTIONS方法",
        "audio_features": None,
        "params": {}
    }).encode('utf-8')
    start_response('405 Method Not Allowed', headers)
    return [response]

# Vercel Serverless入口（WSGI适配）
def handler(event, context):
    from wsgiref.simple_server import make_server
    from io import BytesIO
    
    # 构造WSGI environ
    environ = {
        'REQUEST_METHOD': event['httpMethod'],
        'PATH_INFO': event['path'],
        'QUERY_STRING': event['queryStringParameters'] or '',
        'CONTENT_TYPE': event['headers'].get('content-type', ''),
        'CONTENT_LENGTH': str(len(event['body'] or '')),
        'SERVER_NAME': 'vercel',
        'SERVER_PORT': '80',
        'wsgi.input': BytesIO(event['body'].encode('utf-8') if event.get('body') else b''),
        'wsgi.errors': BytesIO(),
        'wsgi.multithread': False,
        'wsgi.multiprocess': False,
        'wsgi.run_once': False,
        'wsgi.version': (1, 0),
        'wsgi.url_scheme': 'https'
    }
    
    # 添加headers到environ
    for k, v in event['headers'].items():
        key = f'HTTP_{k.replace("-", "_").upper()}'
        environ[key] = v
    
    # 处理响应
    response_body = []
    status = None
    response_headers = {}
    
    def start_response(s, h):
        nonlocal status
        status = s
        for k, v in h:
            response_headers[k] = v
    
    # 执行WSGI应用
    result = application(environ, start_response)
    for chunk in result:
        response_body.append(chunk)
    
    # 构造Vercel响应
    return {
        'statusCode': int(status.split()[0]),
        'headers': response_headers,
        'body': b''.join(response_body).decode('utf-8')
    }

# 本地测试
if __name__ == "__main__":
    from wsgiref.simple_server import make_server
    server = make_server('0.0.0.0', 8000, application)
    print("Local server running on http://0.0.0.0:8000")
    server.serve_forever()
