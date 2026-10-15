"""Convert the IEEE paper source (filled from the template) into Springer LNCS format.  Usage: python paper/springer2/to_lncs.py"""
import re
s = open("paper/ieee/ememoveri_ieee.tex").read()
doc = s[s.index("\\begin{document}"):]
pre = r'''\documentclass[runningheads]{llncs}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\renewcommand\UrlFont{\color{blue!50!black}\rmfamily}
\input{numbers.tex}
\graphicspath{{figures/}}
\setlength{\emergencystretch}{3em}
\newcommand{\sys}{EmemoVeri}
\hyphenation{Ememo-Veri}
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}\setlength{\floatsep}{8pt plus 2pt minus 2pt}\setlength{\intextsep}{8pt plus 2pt minus 2pt}
\renewcommand{\topfraction}{0.9}\renewcommand{\bottomfraction}{0.9}\renewcommand{\textfraction}{0.1}\renewcommand{\floatpagefraction}{0.8}
'''
title = re.search(r"\\title\{(.*?)\}\n\\author", doc, re.S).group(1)
abstract = re.search(r"\\begin\{abstract\}\n(.*?)\n\\end\{abstract\}", doc, re.S).group(1)
body = doc[doc.index("\\section{Introduction}"):]
front = ("\\begin{document}\n\\title{" + title + "}\n\\titlerunning{Retrieving Over Evidence Items}\n\\author{Mihit \\and Priyanka Tyagi}\n\\authorrunning{M. Mihit and P. Tyagi}\n"
         "\\institute{IILM University, Greater Noida, India\\\\\n\\email{mihit.nanda.cs27@iilm.edu}}\n\\maketitle\n\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n"
         "\\keywords{Coding agents \\and Agent memory \\and Repository mining \\and Change localisation \\and Empirical evaluation}\n\n")
body = body.replace("\\begin{figure*}", "\\begin{figure}").replace("\\end{figure*}", "\\end{figure}")
body = body.replace("\\begin{table*}", "\\begin{table}").replace("\\end{table*}", "\\end{table}")
body = body.replace("width=0.62\\textwidth", "width=0.84\\linewidth").replace("width=0.85\\textwidth", "width=\\linewidth")
body = re.sub(r"\\resizebox\{[0-9.]+\\textwidth\}", r"\\resizebox{\\linewidth}", body)
body = body.replace("\\resizebox{\\columnwidth}", "\\resizebox{\\linewidth}").replace("width=\\columnwidth", "width=\\linewidth")
body = body.replace("p{0.52\\columnwidth}", "p{0.55\\linewidth}").replace("p{0.14\\columnwidth}", "p{0.16\\linewidth}")
for t in ("systems", "contrasts", "component"):
    body = body.replace(f"tables/v06_{t}.tex", f"tables/v06_{t}_n.tex")
body = body.replace("; last columns: $\\Delta$Hit@5 on the 10 Python and 6 other-language repositories.", "; the Python and other-language subsets are given in the text.")
body = body.replace("\\bibliography{refs}", "{\\footnotesize\\bibliography{refs}}").replace("\\bibliographystyle{IEEEtran}", "\\bibliographystyle{splncs04}")
open("paper/springer2/ememoveri_lncs.tex", "w").write(pre + front + body)
print("ok")
