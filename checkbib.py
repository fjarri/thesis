import re

auxname = 'thesis.aux'
bibname = 'thesis.bib'

citations_used = set()
with open(auxname) as f:
    for l in f.readlines():
        m = re.match(r'^\\bibcite\{(.*)\}\{\d+\}$', l)
        if m is not None:
            citations_used.add(m.group(1))

citations_bib = set()
with open(bibname) as f:
    for l in f.readlines():
        m = re.match(r'^@\w+\{([^,]+),$', l)
        if m is not None:
            citations_bib.add(m.group(1))

print "Unused citations:", list(sorted(citations_bib - citations_used))
