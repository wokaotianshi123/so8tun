import os
import json
import wave
import struct
import math
import warnings

# 1. 忽略无关警告（pydub语法警告/ffmpeg检测警告）
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 2. 强制指定Vercel内置的ffmpeg路径（关键修复）
os.environ["FFMPEG_PATH"] = "/usr/bin/ffmpeg"
os.environ["PYDUB_FFMPEG_PATH"] = "/usr/bin/ffmpeg"

# 3. 导入依赖（确保starlette/mangum已安装）
import numpy as np
from pydub import AudioSegment
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.requests import Request
from starlette.routing import Route

# 通用音频特征提取（复用之前逻辑，已包含容错）
def extract_audio_features(audio_path):
    try:
        if not audio_path.lower().endswith((".mp3", ".wav")):
            return {"success": False, "error": "仅支持MP3/WAV格式"}
        
        # 强制指定ffmpeg路径
        AudioSegment.converter = "/usr/bin/ffmpeg"
        AudioSegment.ffmpeg = "/usr/bin/ffmpeg"
        
        temp_wav_path = "/tmp/temp_audio.wav"
        try:
            if audio_path.lower().endswith(".mp3"):
                audio = AudioSegment.from_mp3(audio_path)
            else:
                audio = AudioSegment.from_wav(audio_path)
        except Exception as e:
            return {"success": False, "error": f"音频解码失败：{str(e)}（请确保是标准MP3/WAV格式）"}
        
        audio = audio.set_channels(1).set_sample_width(2).set_frame_rate(44100)
        audio.export(temp_wav_path, format="wav", bitrate="128k")

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

# OPTIONS请求处理
async def options_handler(request: Request):
    return JSONResponse({}, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Multipart/form-data"
    })

# 核心接口处理
async def generate_params_handler(request: Request):
    try:
        form_data = await request.form()
        text_params = {
            "mic_model": form_data.get("mic_model", ""),
            "phone_system": form_data.get("phone_system", ""),
            "usage": form_data.get("usage", ""),
            "song_style": form_data.get("song_style", ""),
            "noise_level": form_data.get("noise_level", ""),
            "voice_type": form_data.get("voice_type", ""),
            "voice_problem": form_data.get("voice_problem", "")
        }
        
        audio_file = form_data.get("audio_file")
        audio_features = None
        if audio_file:
            if audio_file.size > 5 * 1024 * 1024:
                return JSONResponse({
                    "code": 400,
                    "msg": "音频文件大小不能超过5MB，请压缩后上传",
                    "audio_features": None,
                    "params": {}
                }, headers={"Access-Control-Allow-Origin": "*"})
            
            temp_path = "/tmp/upload_audio"
            with open(temp_path, "wb") as f:
                f.write(await audio_file.read())
            
            audio_features = extract_audio_features(temp_path)
            os.remove(temp_path)
        
        params = generate_so8_params(text_params, audio_features)
        
        return JSONResponse({
            "code": 200,
            "msg": "参数生成成功",
            "audio_features": audio_features,
            "params": params
        }, headers={"Access-Control-Allow-Origin": "*"})
    except Exception as e:
        return JSONResponse({
            "code": 500,
            "msg": f"服务器内部错误：{str(e)}",
            "audio_features": None,
            "params": {}
        }, headers={"Access-Control-Allow-Origin": "*"})

# 创建应用
app = Starlette(debug=False)
app.add_route("/api/generate_params", generate_params_handler, methods=["POST"])
app.add_route("/api/generate_params", options_handler, methods=["OPTIONS"])

# Vercel入口
def handler(event, context):
    try:
        import mangum
        return mangum.Mangum(app)(event, context)
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({
                "code": 500,
                "msg": f"函数执行失败：{str(e)}",
                "audio_features": None,
                "params": {}
            })
        }

# 本地测试
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
