"""
主程序入口
"""

from api.streamlit_app import StreamlitApp

def main():
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()