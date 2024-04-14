import PyInstaller.__main__

PyInstaller.__main__.run(
    [
        "main.py",
        "--onefile",
        "--console",
        "--icon=assets/app.ico",
        "--add-data=weight;.",
    ]
)
