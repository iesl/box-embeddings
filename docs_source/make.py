import subprocess
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument(
        "-b", "--smv_branch_whitelist", help="Name of the local branch"
    )
    args = parser.parse_args()
    subprocess_args = [
        "sphinx-multiversion",
        ".",
        "../docs",
        "-D",
        "autoapi_dirs=${sourcedir}/../box_embeddings",
        "-D",
        "autoapi_root=${sourcedir}",
    ]

    if args.local:
        if args.smv_branch_whitelist is None:
            raise ValueError(
                "argument smv_branch_whitelist is required if using --local"
            )
        else:
            subprocess_args += [
                "-D",
                f"smv_branch_whitelist={args.smv_branch_whitelist}",
            ]
    subprocess.run(subprocess_args)
    print("Writing root index.html in the docs/")
    redirect_url = (
        "main" if not args.local else args.smv_branch_whitelist
    ) + "/index.html"
    main_index = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <meta http-equiv="Refresh" content="0; url={redirect_url}" />
      </head>
      <body>
        <p>See the latest documentation <a href="{redirect_url}">here</a>.</p>
      </body>
    </html>
    """
    with open("../docs/index.html", "w") as f:
        f.write(main_index)
