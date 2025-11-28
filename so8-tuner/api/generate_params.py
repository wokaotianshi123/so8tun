import os
import json
import tempfile
import numpy as np
import librosa
import scipy
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.requests import Request
from starlette.routing import Route
import uvicorn

# 音频特征提取函数（核心逻辑不变，适配Vercel临时目录）
def extract_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 1. 噪音段特征（前10秒）
        noise_end = min(int(10 * sr), len(y))
        noise_y = y[:noise_end]
        noise_energy = np.sum(np.abs(noise_y)) / len(noise_y) if len(noise_y) > 0 else 0
        noise_spec = librosa.amplitude_to_db(np.abs(librosa.stft(noise_y)), ref=np.max)
        noise_low_freq_energy = np.mean(noise_spec[:10, :]) if len(noise_spec) > 10 else 0
        
        # 2. 人声段特征（10秒后）
        voice_start = max(int(10 * sr), 0)
        voice_end = min(int(60 * sr), len(y))
        voice_y = y[voice_start:voice_end] if voice_end > voice_start else y
        voice_rms = librosa.feature.rms(y=voice_y)[0]
        voice_avg_volume = 20 * np.log10(np.mean(voice_rms) + 1e-8)
        voice_peak_volume = 20 * np.log10(np.max(voice_rms) + 1e-8)
        voice_min_volume = 20 * np.log10(np.min(voice_rms[voice_rms > 0]) + 1e-8) if len(voice_rms[voice_rms > 0]) > 0 else -80
        voice_dynamic_range = voice_peak_volume - voice_min_volume
        
        # 3. 人声频谱特征
        voice_spec = librosa.amplitude_to_db(np.abs(librosa.stft(voice_y)), ref=np.max)
        high_freq_idx = librosa.fft_frequencies(sr=sr) > 5000
        high_freq_energy = np.mean(voice_spec[high_freq_idx, :]) if np.any(high_freq_idx) else -80
        mid_freq_idx = (librosa.fft_frequencies(sr=sr) > 1000) & (librosa.fft_frequencies(sr=sr) < 3000)
        mid_freq_energy = np.mean(voice_spec[mid_freq_idx, :]) if np.any(mid_freq_idx) else -80
        
        # 4. 人声基频
        f0, _ = librosa.pyin(voice_y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_mean = np.nanmean(f0) if not np.all(np.isnan(f0)) else 0
        
        return {
            "duration": duration,
            "noise_energy": noise_energy,
            "noise_low_freq_energy": noise_low_freq_energy,
            "voice_avg_volume": voice_avg_volume,
            "voice_peak_volume": voice_peak_volume,
            "voice_dynamic_range": voice_dynamic_range,
            "high_freq_energy": high_freq_energy,
            "mid_freq_energy": mid_freq_energy,
            "f0_mean": f0_mean,
            "success": True
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# 参数生成函数
def generate_so8_params(form_data, audio_features=None):
    params = {
        "basic_knobs": "",
        "noise_reduction": "-70db",
        "eq_params": "",
        "compressor_params": "",
        "reverb_params": "",
        "optimize_tips": ""
    }
    
    # 1. 基础旋钮
    mic_volume = 24
    record_volume = 25
    if "动圈" in form_data.get("mic_model", ""):
        mic_volume = 26
    elif "电容" in form_data.get("mic_model", ""):
        mic_volume = 22
    if form_data.get("usage") == "sing-live":
        record_volume = 24
    # 音频适配
    if audio_features and audio_features.get("success"):
        if audio_features["voice_peak_volume"] > -10:
            record_volume = max(20, record_volume - 3)
        elif audio_features["voice_peak_volume"] < -25:
            record_volume = min(28, record_volume + 2)
    params["basic_knobs"] = f"话筒音量：{mic_volume} | 耳机音量：25 | 伴奏音量：23 | 录音音量：{record_volume} | 音效音量：18"
    
    # 2. 降噪阈值
    if audio_features and audio_features.get("success"):
        if audio_features["noise_energy"] > 0.1:
            params["noise_reduction"] = "-60db"
        elif audio_features["noise_energy"] < 0.02:
            params["noise_reduction"] = "-82db"
        if audio_features["noise_low_freq_energy"] > -40:
            params["noise_reduction"] = f"{int(params['noise_reduction'].replace('db', '')) - 5}db"
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
    
    # 3. 均衡器
    eq_base = "q值统一4.5 | 高通12档(40hz) | 低通12档(12000hz) | "
    eq_custom = ""
    if audio_features and audio_features.get("success"):
        if audio_features["high_freq_energy"] > -20:
            eq_custom += "5350hz增益-2.5 | 9500hz增益-2 | "
        elif audio_features["high_freq_energy"] < -40:
            eq_custom += "5350hz增益+3 | 9500hz增益+2.5 | "
        if audio_features["mid_freq_energy"] > -15:
            eq_custom += "1350hz增益-3 | "
    if "有鼻音" in form_data.get("voice_problem", ""):
        eq_custom += "1350hz增益-2.5 | "
    if form_data.get("voice_type") == "thin":
        eq_custom += "100hz增益+3.5 | 450hz增益+2.5 | "
    if form_data.get("voice_type") == "thick":
        eq_custom += "100hz增益-1.5 | 280hz增益-1.5 | "
    params["eq_params"] = eq_base + eq_custom + "100hz增益+2 | 140hz增益-1 | 280hz增益0 | 450hz增益+1.5 | 850hz增益-0.5 | 2800hz增益+1 | 10000hz增益+1.5"
    
    # 4. 压缩器
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
    
    # 5. 混响
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
    
    # 6. 优化建议
    tips = []
    if audio_features and audio_features.get("success"):
        if audio_features["voice_peak_volume"] > -5:
            tips.append("1. 人声峰值音量较高，建议录音音量再降低2-3，避免爆音；")
        elif audio_features["voice_avg_volume"] < -30:
            tips.append("1. 人声平均音量较低，建议话筒音量提高2-3，压缩增益补偿+4；")
        if audio_features["noise_energy"] > 0.15:
            tips.append("2. 噪音能量较高，建议增加吸音棉/关闭门窗，降噪阈值可微调至-55db；")
        if audio_features["high_freq_energy"] > -20:
            tips.append("3. 人声高频过强，若仍刺耳可将5350hz增益再减1；")
        elif audio_features["high_freq_energy"] < -40:
            tips.append("3. 人声高频不足，若声音发闷可将9500hz增益再+1；")
    if not tips:
        tips.append("1. 若唱歌爆音，可将录音音量再降低1-2；2. 若混响盖过人声，可降低混响音量或干湿比；")
    if "鼻音" in form_data.get("voice_problem", ""):
        tips.append("4. 若鼻音仍明显，可将1350hz增益再减1；")
    params["optimize_tips"] = "".join(tips)
    
    return params

# 处理OPTIONS请求（跨域预检）
async def options_handler(request: Request):
    return JSONResponse({}, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Multipart/form-data"
    })

# 核心接口处理函数
async def generate_params_handler(request: Request):
    try:
        # 解析表单数据
        form_data = await request.form()
        # 提取文本参数
        text_params = {
            "mic_model": form_data.get("mic_model", ""),
            "phone_system": form_data.get("phone_system", ""),
            "usage": form_data.get("usage", ""),
            "song_style": form_data.get("song_style", ""),
            "noise_level": form_data.get("noise_level", ""),
            "voice_type": form_data.get("voice_type", ""),
            "voice_problem": form_data.get("voice_problem", "")
        }
        
        # 处理音频文件
        audio_file = form_data.get("audio_file")
        audio_features = None
        if audio_file:
            # 保存到Vercel临时目录
            temp_path = os.path.join("/tmp", audio_file.filename)
            with open(temp_path, "wb") as f:
                f.write(await audio_file.read())
            # 提取特征
            audio_features = extract_audio_features(temp_path)
            # 删除临时文件
            os.remove(temp_path)
        
        # 生成参数
        params = generate_so8_params(text_params, audio_features)
        
        return JSONResponse({
            "code": 200,
            "msg": "参数生成成功",
            "audio_features": audio_features,
            "params": params
        }, headers={
            "Access-Control-Allow-Origin": "*"
        })
    except Exception as e:
        return JSONResponse({
            "code": 500,
            "msg": f"参数生成失败：{str(e)}",
            "audio_features": None,
            "params": {}
        }, headers={
            "Access-Control-Allow-Origin": "*"
        })

# 创建Starlette应用
app = Starlette(debug=True, routes=[
    Route("/api/generate_params", generate_params_handler, methods=["POST"]),
    Route("/api/generate_params", options_handler, methods=["OPTIONS"])
])

# Vercel Serverless入口
def handler(event, context):
    import mangum
    return mangum.Mangum(app)(event, context)

# 本地测试入口
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)