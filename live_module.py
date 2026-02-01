import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP
def run_live_sniffing():
    # هدر با افکت نئونی
    st.markdown("""
        <h1 style='text-align: center; color: #FF4B4B; text-shadow: 0px 0px 10px #FF4B4B;'>
            🎯 ACTIVE THREAT INTERCEPTOR
        </h1>
        <p style='text-align: center; color: #888;'>Real-time Packet Analysis & Deep Stream Inspection</p>
    """, unsafe_allow_html=True)

    # بخش تنظیمات با طراحی کارتی
    # بخش تنظیمات با طراحی کارتی اصلاح شده
    with st.expander("🛠️ INTERFACE CONFIGURATION", expanded=True):
        col_c1, col_c2, col_c3 = st.columns([2, 2, 1])
        with col_c1:
            iface = st.text_input("🎯 Target Interface", "Wi-Fi", help="Adapter Name")
        with col_c2:
            # تغییر از select_slider به number_input برای آزادی در انتخاب هر عددی
            pkt_count = st.number_input(
                "📦 Capture Volume (Packet Count)",
                min_value=1,
                max_value=10000,
                value=100,
                step=10,
                help="هر عددی بین ۱ تا ۱۰,۰۰۰ وارد کنید"
            )
        with col_c3:
            st.write("##")
            btn_live = st.button("⚡ INITIALIZE", use_container_width=True)

    if btn_live:
        metric_holder = st.empty()
        chart_container = st.container()
        log_header = st.empty()
        log_holder = st.empty()

        try:
            with st.status("🚀 Sniffing Network Packets...", expanded=True) as status:
                st.write("Accessing Raw Sockets...")
                packets = sniff(iface=iface, count=pkt_count, timeout=20)
                st.write("Processing Data Streams...")
                status.update(label="✅ Scan Complete!", state="complete", expanded=False)

            if len(packets) == 0:
                st.error("📡 Signal Lost: No packets captured. Try Running as Admin.")
            else:
                ip_pkts = [p for p in packets if IP in p]
                tcp_c = sum(1 for p in ip_pkts if p.haslayer(TCP))
                udp_c = sum(1 for p in ip_pkts if p.haslayer(UDP))
                others = len(packets) - (tcp_c + udp_c)
                risk_status = "CRITICAL" if len(packets) > (pkt_count * 0.7) else "STABLE"

                # ۱. نمایش کارت‌های شاخص فوق حرفه‌ای
                # در بخش نمایش متریک‌های زنده
                with metric_holder.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Captured", len(packets))
                    m2.metric("TCP/UDP Mix", f"{tcp_c}/{udp_c}")

                    # روانشناسی: نمایش ریسک سیستم با رنگ قرمز در صورت خطر
                    risk_color = "inverse" if risk_status == "CRITICAL" else "normal"

                    m3.metric(
                        label="System Risk",
                        value=risk_status,
                        delta="Action Required" if risk_status == "CRITICAL" else "Stable",
                        delta_color=risk_color  # <--- اینجا تغییر اعمال می‌شود
                    )
                # ۲. بخش بصری‌سازی پیشرفته
                with chart_container:
                    col_g1, col_g2 = st.columns(2)

                    with col_g1:
                        fig_g = go.Figure(go.Indicator(
                            mode="gauge+number", value=len(packets),
                            gauge={'axis': {'range': [0, pkt_count]},
                                   'bar': {'color': "#FF4B4B" if risk_status == "CRITICAL" else "#00D084"},
                                   'steps': [{'range': [0, pkt_count / 2], 'color': "#111"},
                                             {'range': [pkt_count / 2, pkt_count], 'color': "#222"}]},
                            title={'text': "Traffic Intensity", 'font': {'color': '#FF4B4B'}}
                        ))
                        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=300)
                        st.plotly_chart(fig_g, use_container_width=True)

                    with col_g2:
                        # نمودار توزیع پروتکل با افکت Donut
                        fig_pie = px.pie(values=[tcp_c, udp_c, others], names=['TCP', 'UDP', 'Other'],
                                         hole=0.6, color_discrete_sequence=['#58A6FF', '#FF4B4B', '#FFD166'])
                        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', showlegend=True, height=300,
                                              legend=dict(font=dict(color="white")))
                        st.plotly_chart(fig_pie, use_container_width=True)

                # ۳. شبیه‌ساز ترمینال برای لاگ‌ها
                log_header.markdown("#### ⌨️ TERMINAL OUTPUT")
                live_logs = [
                    f"PROTO: {'TCP' if p.haslayer(TCP) else 'UDP' if p.haslayer(UDP) else 'IP'} | {p[IP].src} >> {p[IP].dst} | LEN: {len(p)}"
                    for p in ip_pkts]

                log_style = """
                <style>
                .terminal-box {
                    background-color: #000;
                    color: #00FF41;
                    padding: 15px;
                    border-radius: 5px;
                    font-family: 'Courier New', Courier, monospace;
                    border: 1px solid #333;
                    max-height: 300px;
                    overflow-y: scroll;
                }
                </style>
                """
                st.markdown(log_style, unsafe_allow_html=True)
                log_content = "".join([f"<div>$ {log}</div>" for log in live_logs[:50]])
                log_holder.markdown(f'<div class="terminal-box">{log_content}</div>', unsafe_allow_html=True)

                # ۴. گزارش نهایی در قالب کارت شناسایی
                st.markdown("---")
                top_ip = pd.Series([p[IP].src for p in ip_pkts]).mode()[0] if ip_pkts else "N/A"

                rep_c1, rep_c2 = st.columns([3, 1])
                with rep_c1:
                    st.info(
                        f"🛡️ **Security Summary:** Interface **{iface}** is currently **{risk_status}**. Most active node: `{top_ip}`")
                with rep_c2:
                    report_txt = f"SECURITY AUDIT\nDate: {time.strftime('%Y-%m-%d %H:%M:%S')}\nRisk: {risk_status}\nPackets: {len(packets)}"
                    st.download_button("📥 EXPORT REPORT", report_txt, file_name=f"Live_Audit_{int(time.time())}.txt",
                                       use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ SYSTEM FAULT: {str(e)}")