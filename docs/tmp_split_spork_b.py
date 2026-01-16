from dataclasses import dataclass

for i in range(3):
    input(f"Press enter {i+1}/3 times to overwrite outputs...")

input_lines = open("spork_b_old.tex").read().split("\n")
num_lines = len(input_lines)

files_dict = {}

def begin_file(name):
    assert name not in files_dict
    line_objs = []
    files_dict[name] = line_objs
    return line_objs

@dataclass(slots=True)
class SectionLine:
    text: str
    filename: str

@dataclass(slots=True)
class SubsectionLine:
    text: str
    filename: str

@dataclass(slots=True)
class GeneratedInputLine:
    text: str

@dataclass(slots=True)
class EndDocumentLine:
    text: str

@dataclass(slots=True)
class NormalLine:
    text: str

def parse_magic(line):
    parts = line.split("}{")
    assert len(parts) == 2
    end = parts[1]
    assert end[-1] == "}"
    assert end[:4] == "sec:"
    text = line
    text = text.replace("{b_samples/", "{spork_b/")
    filename = "spork_b/" + end[4:-1] + ".tex"
    return line, filename


def parse_line(line):
    if line.startswith("\\magicSection"):
        return SectionLine(*parse_magic(line))
    if line.startswith("\\magicSubsection"):
        return SubsectionLine(*parse_magic(line))
    assert "\\magicSection" not in line
    assert "\\magicSubection" not in line
    if line == "\\end{document}":
        return EndDocumentLine(line)
    return NormalLine(line)

def process_main():
    i = 0
    line_objs = begin_file("spork_b/main.tex")
    while i < num_lines:
        line_obj = parse_line(input_lines[i])
        if isinstance(line_obj, SectionLine):
            line_objs.append(GeneratedInputLine("\\input{%s}" % line_obj.filename))
            i = process_section(i)
        else:
            line_objs.append(line_obj)
            i += 1
        assert not isinstance(line_obj, SubsectionLine)

def process_section(i):
    line_obj = parse_line(input_lines[i])
    assert isinstance(line_obj, SectionLine)
    i += 1
    line_objs = begin_file(line_obj.filename)
    line_objs.append(line_obj)
    while i < num_lines:
        line_obj = parse_line(input_lines[i])
        if isinstance(line_obj, (SectionLine, EndDocumentLine)):
            return i
        elif isinstance(line_obj, SubsectionLine):
            line_objs.append(GeneratedInputLine("\\input{%s}" % line_obj.filename))
            i = process_subsection(i)
        else:
            line_objs.append(line_obj)
            i += 1
    return i

def process_subsection(i):
    line_obj = parse_line(input_lines[i])
    assert isinstance(line_obj, SubsectionLine)
    i += 1
    line_objs = begin_file(line_obj.filename)
    line_objs.append(line_obj)
    while i < num_lines:
        line_obj = parse_line(input_lines[i])
        if isinstance(line_obj, (SectionLine, SubsectionLine, EndDocumentLine)):
            return i
        else:
            line_objs.append(line_obj)
            i += 1
    return i


process_main()


for filename, line_objs in files_dict.items():
    with open(f"{filename}", "w") as f:
        fig_begin = 0
        for i, line_obj in enumerate(line_objs):
            txt = line_obj.text
            f.write(txt)
            f.write("\n")
            if "\\begin{figure}" in txt:
                fig_begin = i
            if "\\end{figure}" in txt:
                print(f"Figure at {filename}:{i+1} ({i-fig_begin} lines)")
            if "\\input" in txt and not isinstance(line_obj, GeneratedInputLine):
                pass
                # print(f"Input at {filename}:{i+1}")


open("spork_b.tex", "w").write("\\input{spork_b/main.tex}\n")
