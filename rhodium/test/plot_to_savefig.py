import sys
import re
import pathlib

if __name__ == "__main__":    
    if len(sys.argv) != 3:
        print("Usage: " + __file__ + " [file] [fig_path]")
        sys.exit(-1)

    fig_index = 1
    fig_basename = pathlib.Path(sys.argv[2]) / pathlib.Path(sys.argv[1]).stem
    pyplot_alias = "matplotlib.pyplot"
    result = []

    with open(sys.argv[1], "r") as f:
        for line in f:
            m = re.match(r"import matplotlib.pyplot as ([a-zA-Z_]+)", line)
            if m:
                pyplot_alias = m.group(1)

            m = re.match(pyplot_alias + r".show\(\)", line)
            if m:
                line = line[:m.start()] + pyplot_alias + ".savefig('" + str(fig_basename) + "." + str(fig_index) + ".png')" + line[m.end():]
                fig_index += 1

            result.append(line)

    with open(sys.argv[1], "w") as f:
        for line in result:
            f.write(line)
