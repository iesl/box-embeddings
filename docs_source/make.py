import subprocess
import os

if __name__ == "__main__":
    subprocess.run(
        [
            "sphinx-multiversion",
            ".",
            "../docs",
            "-D",
            "autoapi_dirs=${sourcedir}/../box_embeddings",
            "-D",
            "autoapi_root=${sourcedir}",
        ]
    )
    main_index = """
    <!DOCTYPE html>
    <html>
      <head>
        <meta http-equiv="Refresh" content="0; url=main/index.html" />
      </head>
      <body>
        <p>See the latest documentation <a href="main/index.html">here</a>.</p>
      </body>
    </html>
    """
    with open('../docs/index.html', 'w') as f:
        f.write(main_index)

