import re
import os, os.path


def load_tex(fname):
    with open(fname) as f:
        contents = f.read()

    includes = re.findall(r'\\(include|input)\{([^\}]+)\}', contents)
    for include in includes:
        if include[1] in ('titlepage', 'declaration'):
        # they are too tex-heavy, better skip them altogether
            sub_contents = '\n'
        else:
            sub_contents = load_tex(include[1] + '.tex')
        contents = contents.replace(
            '\\' + include[0] + '{' + include[1] + '}',
            '\n' + sub_contents + '\n')

    return contents

def strip_preamble(text):
    begin_mark = '\\begin{document}'
    end_mark = '\\end{document}'
    begin_idx = text.find(begin_mark)
    end_idx = text.find(end_mark)
    return text[begin_idx + len(begin_mark):end_idx]

def strip_envs(text, env):
    return re.sub(r'\\begin\{' + env + r'\}(.*?)\\end\{' + env + r'\}',
        '',
        text,
        flags=re.DOTALL)

def strip_comments(text):
    lines = text.split('\n')
    lines = [line for line in lines if not line.startswith('%')]
    return '\n'.join(lines)

def replace_display_eqn(text, env):
    def replace(m):
        s = m.group(1).strip()
        if s.endswith('\\qedhere'):
            s = s[-8:].strip()
        if s[-1] == '.':
            return ' (this equation). '
        elif s[-1] == ',':
            return ' (this equation), '
        else:
            return ' (this equation) '

    return re.sub(r'\n[ \t]*?\\begin\{' + env + r'\}(.*?)\\end\{' + env + r'\}\n',
        replace,
        text,
        flags=re.DOTALL)

def replace_inline_eqn(text):
    return re.sub(r'\$[^\$]+\$', 'this equation', text)

def strip_commands(text, cmd):
    return re.sub(r'\~?\\' + cmd + r'\{[^\}]+\}', '', text)

def expand_abbrevs(text):
    return re.sub(r'\\abbrev\{([^\}]+)\}', lambda m: m.group(1).upper(), text)

def expand_command(text, cmd):
    return re.sub(r'\\' + cmd + r'\{([^\}]+)\}', lambda m: m.group(1), text)

def expand_ref(text, ref, subs):
    return re.sub(r'\~?\\' + ref + r'\{[^\}]+\}', ' ' + subs, text)

def expand_envs(text, env):
    return re.sub(r'\\begin\{' + env + r'\}(\[.*?\])?(.*?)\\end\{' + env + r'\}',
        lambda m: m.group(2),
        text,
        flags=re.DOTALL)

def replace_figures(text):
    return re.sub(r'\\begin\{figure\}(.*?)\\end\{figure\}',
        lambda m: re.sub(
            r'^.*\\caption\[[^\]]+\]\{(.*)\}\%endcaption.*$',
            r'\1',
            m.group(1),
            flags=re.DOTALL),
        text,
        flags=re.DOTALL)

def replace_itemizes(text, env):
    return re.sub(r'\\begin\{' + env + r'\}(.*?)\\end\{' + env + r'\}',
        lambda m: re.sub(
            r'\\item(\[[^\]]\])?',
            '',
            m.group(1),
            flags=re.DOTALL),
        text,
        flags=re.DOTALL)



if __name__ == '__main__':
    text = load_tex('thesis.tex')
    text = strip_preamble(text)
    text = strip_comments(text)
    text = strip_envs(text, 'spacing')
    text = replace_display_eqn(text, 'eqn')
    text = replace_display_eqn(text, 'eqn\\*')
    text = replace_display_eqn(text, 'eqn2')
    text = replace_display_eqn(text, 'eqns')
    text = replace_inline_eqn(text)
    text = strip_commands(text, 'label')
    text = strip_commands(text, 'cite')
    text = strip_commands(text, 'centerline')
    text = expand_abbrevs(text)
    text = expand_command(text, 'chapter')
    text = expand_command(text, 'section')
    text = expand_command(text, 'subsection')
    text = expand_command(text, 'addchap\\*')
    text = expand_command(text, 'textbf')
    text = expand_command(text, 'textit')
    text = expand_ref(text, 'defref', 'that definition')
    text = expand_ref(text, 'thmref', 'that theorem')
    text = expand_ref(text, 'lmmref', 'that lemma')
    text = expand_ref(text, 'charef', 'that chapter')
    text = expand_ref(text, 'secref', 'that section')
    text = expand_ref(text, 'appref', 'that appendix')
    text = expand_ref(text, 'figref', 'that figure')
    text = expand_ref(text, 'eqnref', 'that equation')
    text = expand_envs(text, 'proof')
    text = expand_envs(text, 'definition')
    text = expand_envs(text, 'theorem')
    text = expand_envs(text, 'lemma')
    text = replace_itemizes(text, 'itemize')
    text = replace_itemizes(text, 'enumerate')
    text = replace_figures(text)
    text = re.sub(r'(\\\^|\\"|\\\')', '', text)
    text = text.replace('{\\o}', 'o')
    text = text.replace('\\,', ' ')
    text = text.replace('~', ' ')
    text = text.replace('\\Rb{}', 'Rubidium')
    text = text.replace('\\cleardoublepage', '')
    text = text.replace('\\frontmatter', '')
    text = text.replace('\\normalheaders', '')
    text = text.replace('\\mainmatter', '')
    text = text.replace('\\appendix', '')
    text = text.replace(r'\renewcommand{\thechapter}{\alph{chapter}}', '')
    text = re.sub(r'\\counterwithin\{\w+\}\{\w+\}', '', text)
    print text
