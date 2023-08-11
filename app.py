import streamlit as st
# import anthropic
import pandas as pd
from scipy.io import wavfile


# whiper
import gradio as gr
from faster_whisper import WhisperModel

# ファイルを一時的に保存する
from tempfile import NamedTemporaryFile

# 音声再生
import base64
import time
import io

# whisperの処理
model_size = "tiny"
model = WhisperModel(model_size, device="cpu", compute_type="int8")


with st.sidebar:
    st.markdown("""
      ## 概要
      音声文字起こしアプリです。

      ### 使い方
     1. mp3/mp4ファイルをアップロードします。
     2. 処理された結果が下に表示されます。
    """)


st.title("🎙️🎙️🎙️文字起こしアプリ🎙️🎙️🎙️")
uploaded_file = st.file_uploader("Upload an article", type=["mp4","mp3"])
st.info("📝処理が終われば、以下に結果が表示されます")


result = pd.DataFrame(
        {'start': [], 'end': [], 'text': []},
        index=[])

def spacing():
    st.markdown("<br></br>", unsafe_allow_html=True)

if uploaded_file is not None:
  # デフォルトのtranscribeはファイル名を入力にする。そのためアップロードしたファイルを一時的に保存する
    with NamedTemporaryFile(suffix="mp3") as temp:
        temp.write(uploaded_file.getvalue())
        temp.seek(0)
        # st.audio(uploaded_file.getvalue(),start_time=2)
        segments,info  = model.transcribe(temp.name,language='ja',beam_size=5)
        for segment in segments:
            result.loc[str(segment.id)] = [segment.start,segment.end,segment.text]


    for i in range(len(result)):
        st.markdown(
            f"""
            <h4 style='text-align: center; color: black;'>{result['start'][i]}-{result['end'][i]}</h4>
            """,
            unsafe_allow_html=True,
        )
        spacing()
        st.markdown(
            f"""
            <h5 style='text-align: center; color: black;'>{result['text'][i]}</h4>
            """,
            unsafe_allow_html=True,
        )
        st.audio(uploaded_file.getvalue(),start_time=int(result['start'][i]))
        st.markdown("---")
    
    


st.table(result)





# st.markdown(result.to_html(escape=False), unsafe_allow_html=True)


# audio_placeholder = st.empty()
# if uploaded_file is not None:
#     audio_str = "data:audio/ogg;base64,%s"%(base64.b64encode(uploaded_file.getvalue()).decode())
#     audio_html = """
#                     <audio controls>
#                     <source src="%s" type="audio/ogg">
#                     Your browser does not support the audio element.
#                     </audio>
#                 """ %audio_str
#     audio_placeholder.empty()
#     time.sleep(0.5) #これがないと上手く再生されません
#     audio_placeholder.markdown(audio_html, unsafe_allow_html=True)











# def transcribe(audio):

#     # load audio and pad/trim it to fit 30 seconds
#     audio = whisper.load_audio(audio)
#     audio = whisper.pad_or_trim(audio)

#     # make log-Mel spectrogram and move to the same device as the model
#     mel = whisper.log_mel_spectrogram(audio).to(model.device)

#     # detect the spoken language
#     _, probs = model.detect_language(mel)
#     print(f"Detected language: {max(probs, key=probs.get)}")

#     # decode the audio
#     options = whisper.DecodingOptions()
#     result = whisper.decode(model, mel, options)
#     return result.text