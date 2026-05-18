import os
import io
import json
import time
import numpy as np
import streamlit as st
from PIL import Image
import paho.mqtt.client as paho

st.set_page_config(
    page_title="EAFITPark",
    page_icon="🅿️",
    layout="wide"
)

# ---------- STYLE ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #eef4ff 0%, #f7fbff 100%);
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1300px;
}
.card {
    background: white;
    border: 1px solid #d8e6ff;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(50, 90, 160, 0.08);
    margin-bottom: 16px;
}
.hero {
    background: linear-gradient(135deg, #163b73 0%, #2457a7 100%);
    color: white;
    border-radius: 24px;
    padding: 24px;
    margin-bottom: 18px;
    box-shadow: 0 14px 34px rgba(20, 50, 110, 0.22);
}
.hero h1 {
    color: white;
    margin: 0;
}
.hero p {
    color: #dbe8ff;
    margin-top: 8px;
}
.slot-box {
    border-radius: 18px;
    padding: 16px;
    text-align: center;
    border: 2px dashed #b9d0ff;
    min-height: 220px;
    background: #f7fbff;
}
.slot-free {
    background: #f3fff6;
    border: 2px dashed #8dd5a1;
}
.slot-full {
    background: #fff4f4;
    border: 2px dashed #e49a9a;
}
.slot-title {
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 10px;
}
.slot-state-free {
    color: #15803d;
    font-weight: 700;
}
.slot-state-full {
    color: #b91c1c;
    font-weight: 700;
}
.car-emoji {
    font-size: 4rem;
    margin: 14px 0;
}
.park-emoji {
    font-size: 3.2rem;
    margin: 18px 0;
}
.status-pill {
    display: inline-block;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.9rem;
}
.status-full {
    background: #fee2e2;
    color: #b91c1c;
}
.status-available {
    background: #dcfce7;
    color: #15803d;
}
.small-note {
    color: #45608c;
}
</style>
""", unsafe_allow_html=True)

# ---------- STATE ----------
if "sim_slots" not in st.session_state:
    st.session_state.sim_slots = [False, False, False]

if "mqtt_message" not in st.session_state:
    st.session_state.mqtt_message = "Aún no se ha enviado ningún estado."

if "real_result" not in st.session_state:
    st.session_state.real_result = None


# ---------- HELPERS ----------
def on_publish(client, userdata, result):
    pass

def publish_status_to_wokwi(is_full, broker, port, topic):
    """
    ON  = parqueadero lleno  -> LED rojo
    OFF = hay al menos un cupo -> LED verde
    """
    payload = json.dumps({"Act1": "ON" if is_full else "OFF"})

    try:
        client = paho.Client(client_id=f"EAFITPark_{int(time.time())}")
        client.on_publish = on_publish
        client.connect(broker, int(port))
        client.publish(topic, payload)
        client.disconnect()

        status_text = "LLENO -> rojo encendido" if is_full else "DISPONIBLE -> verde encendido"
        return True, f"Estado enviado a Wokwi: {status_text}"
    except Exception as e:
        return False, f"Error enviando a Wokwi: {e}"


@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov5su.pt")
        return model, None
    except Exception as e:
        return None, str(e)


def count_cars_in_image(pil_img, conf_threshold):
    model, error = load_model()
    if model is None:
        raise RuntimeError(error)

    np_img = np.array(pil_img.convert("RGB"))[:, :, ::-1]  # RGB -> BGR

    results = model(
        np_img,
        conf=conf_threshold,
        iou=0.45,
        max_det=300
    )

    result = results[0]
    boxes = result.boxes
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB

    car_count = 0
    details = []

    if boxes is not None and len(boxes) > 0:
        names = model.names

        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            label = names[cls_id] if isinstance(names, dict) else names[cls_id]

            # Solo cuenta carros
            if label == "car":
                car_count += 1
                details.append({
                    "Clase": label,
                    "Confianza": round(conf, 2)
                })

    return annotated, car_count, details


def simulation_status():
    occupied = sum(st.session_state.sim_slots)
    total = len(st.session_state.sim_slots)
    available = total - occupied
    is_full = occupied >= total
    return occupied, total, available, is_full


# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Conexión Wokwi / MQTT")
    broker = st.text_input("Broker", value="157.230.214.127")
    port = st.number_input("Puerto", min_value=1, max_value=65535, value=1883)
    topic = st.text_input("Tópico", value="cmqtt_s")
    st.caption("ON = lleno / OFF = disponible")

    st.divider()

    st.header("YOLO")
    conf_threshold = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)


# ---------- HEADER ----------
st.markdown("""
<div class="hero">
    <h1>EAFITPark</h1>
    <p>
        Simulación de parqueadero + conteo real de carros con cámara + respuesta en Wokwi.
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Simulación", "Parqueadero real"])


# ---------- TAB 1: SIMULATION ----------
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Simulación del parqueadero")
    st.write("Haz clic en cada puesto para parquear o retirar un carro. Cuando los 3 puestos estén ocupados, se enviará estado de lleno a Wokwi.")

    occupied, total, available, is_full = simulation_status()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Puestos ocupados", occupied)
    c2.metric("Puestos libres", available)
    c3.metric("Capacidad total", total)
    with c4:
        if is_full:
            st.markdown("<span class='status-pill status-full'>LLENO</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='status-pill status-available'>DISPONIBLE</span>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(3)

    for i, col in enumerate(cols):
        with col:
            occupied_here = st.session_state.sim_slots[i]

            box_class = "slot-box slot-full" if occupied_here else "slot-box slot-free"
            state_class = "slot-state-full" if occupied_here else "slot-state-free"
            state_text = "Ocupado" if occupied_here else "Libre"

            st.markdown(
                f"""
                <div class="{box_class}">
                    <div class="slot-title">Puesto {i+1}</div>
                    <div class="{state_class}">{state_text}</div>
                """,
                unsafe_allow_html=True
            )

            img_path = f"car{i+1}.png"

            if occupied_here:
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.markdown("<div class='car-emoji'>🚗</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='park-emoji'>🅿️</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            button_label = "Retirar carro" if occupied_here else "Parquear carro"
            if st.button(button_label, key=f"slot_{i}", use_container_width=True):
                st.session_state.sim_slots[i] = not st.session_state.sim_slots[i]

                occupied_now, total_now, available_now, is_full_now = simulation_status()
                ok, msg = publish_status_to_wokwi(is_full_now, broker, port, topic)
                st.session_state.mqtt_message = msg
                st.rerun()

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Vaciar parqueadero", use_container_width=True):
            st.session_state.sim_slots = [False, False, False]
            ok, msg = publish_status_to_wokwi(False, broker, port, topic)
            st.session_state.mqtt_message = msg
            st.rerun()

    with b2:
        if st.button("Llenar parqueadero", use_container_width=True):
            st.session_state.sim_slots = [True, True, True]
            ok, msg = publish_status_to_wokwi(True, broker, port, topic)
            st.session_state.mqtt_message = msg
            st.rerun()

    st.info(st.session_state.mqtt_message)
    st.caption("Si subes archivos llamados car1.png, car2.png y car3.png al proyecto, se usarán en vez del emoji del carro.")
    st.markdown('</div>', unsafe_allow_html=True)


# ---------- TAB 2: REAL PARKING ----------
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Parqueadero real con cámara")
    st.write("Toma una foto del parqueadero o sube una imagen. La app cuenta cuántos carros detecta y decide si está lleno o si queda al menos un cupo.")

    total_spaces = st.number_input("Número total de espacios disponibles", min_value=1, value=3, step=1)

    col_cam, col_upload = st.columns(2)
    with col_cam:
        picture = st.camera_input("Tomar foto del parqueadero")
    with col_upload:
        uploaded_file = st.file_uploader(
            "O subir una imagen",
            type=["png", "jpg", "jpeg"]
        )

    if st.button("Analizar imagen y actualizar Wokwi", use_container_width=True):
        image_source = None

        if picture is not None:
            image_source = io.BytesIO(picture.getvalue())
        elif uploaded_file is not None:
            image_source = uploaded_file
        else:
            st.warning("Primero toma una foto o sube una imagen.")

        if image_source is not None:
            try:
                pil_img = Image.open(image_source).convert("RGB")

                with st.spinner("Detectando carros..."):
                    annotated, detected_cars, details = count_cars_in_image(pil_img, conf_threshold)

                free_spaces = max(int(total_spaces) - detected_cars, 0)
                is_full_real = detected_cars >= int(total_spaces)

                ok, msg = publish_status_to_wokwi(is_full_real, broker, port, topic)

                st.session_state.real_result = {
                    "annotated": annotated,
                    "detected_cars": detected_cars,
                    "total_spaces": int(total_spaces),
                    "free_spaces": free_spaces,
                    "is_full": is_full_real,
                    "details": details,
                    "mqtt_message": msg
                }

            except Exception as e:
                st.error(f"Error durante el análisis: {e}")

    if st.session_state.real_result is not None:
        result = st.session_state.real_result

        col1, col2 = st.columns([1.3, 1])

        with col1:
            st.image(result["annotated"], caption="Imagen analizada", use_container_width=True)

        with col2:
            st.metric("Carros detectados", result["detected_cars"])
            st.metric("Capacidad configurada", result["total_spaces"])
            st.metric("Espacios libres", result["free_spaces"])

            if result["is_full"]:
                st.error("Parqueadero lleno -> se envía ON a Wokwi (LED rojo)")
            else:
                st.success("Hay al menos un espacio libre -> se envía OFF a Wokwi (LED verde)")

            st.info(result["mqtt_message"])

        if result["details"]:
            st.write("Detecciones contadas como carro:")
            st.dataframe(result["details"], use_container_width=True)
        else:
            st.write("No se detectaron carros.")

    st.markdown('</div>', unsafe_allow_html=True)
