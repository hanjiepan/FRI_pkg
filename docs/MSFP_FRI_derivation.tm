<TeXmacs|1.99.6>

<style|generic>

<\body>
  <\equation*>
    x<around*|(|t|)>=<big|sum><rsub|k=1><rsup|K>\<alpha\><rsub|k>\<delta\><around*|(|t-t<rsub|k>|)><space|1em>t\<in\><around*|[|0,\<tau\>|)>
  </equation*>

  <\equation*>
    sinc<around*|(|t|)>=<frac|sin<around*|(|t|)>|t>
  </equation*>

  <\eqnarray*>
    <tformat|<table|<row|<cell|y<rsub|\<ell\>>>|<cell|=>|<cell|<around*|\<langle\>|x<around*|(|t|)>,sinc<around*|(|\<pi\>B<around*|(|t<rprime|'><rsub|\<ell\>>-t|)>|)>|\<rangle\>><rsub|t>>>|<row|<cell|>|<cell|=>|<cell|<big|int>x<around*|(|t|)>sinc<around*|(|\<pi\>B<around*|(|t<rprime|'><rsub|\<ell\>>-t|)>|)>\<mathd\>t>>|<row|<cell|>|<cell|=>|<cell|<big|int><big|sum><rsub|m\<in\><with|math-font|Bbb*|Z>><wide|x|^><rsub|m>\<mathe\><rsup|j2\<pi\>m
    t/\<tau\>>sinc<around*|(|\<pi\>B<around*|(|t<rprime|'><rsub|\<ell\>>-t|)>|)>\<mathd\>t>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|m\<in\><with|math-font|Bbb*|Z>><wide|x|^><rsub|m><big|int>sinc<around*|(|\<pi\>B<around*|(|t<rprime|'><rsub|\<ell\>>-t|)>|)>\<mathe\><rsup|j2\<pi\>m
    t/\<tau\>>\<mathd\>t>>|<row|<cell|change of
    variable>|<cell|=>|<cell|<big|sum><rsub|m\<in\><with|math-font|Bbb*|Z>><wide|x|^><rsub|m>\<mathe\><rsup|j2\<pi\>m
    t<rprime|'><rsub|\<ell\>>/\<tau\>><wide*|<big|int>sinc<around*|(|\<pi\>B
    t|)>\<mathe\><rsup|-j2\<pi\>m t/\<tau\>>\<mathd\>t|\<wide-underbrace\>><rsub|FT
    of sinc<around*|(|\<pi\>B t|)> at \<omega\>=2\<pi\>m/\<tau\>>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|m\<in\><with|math-font|Bbb*|Z>><wide|x|^><rsub|m>\<mathe\><rsup|j2\<pi\>m
    t<rprime|'><rsub|\<ell\>>/\<tau\>>\<cdot\><frac|1|B>rect<around*|(|<frac|2\<pi\>m/\<tau\>|2\<pi\>B>|)>>>|<row|<cell|>|<cell|=>|<cell|<frac|1|B><big|sum><rsub|<around*|\||m|\|>\<leqslant\><around*|\<lfloor\>|B\<tau\>/2|\<rfloor\>>><wide|x|^><rsub|m>\<mathe\><rsup|j2\<pi\>m
    t<rprime|'><rsub|\<ell\>>/\<tau\>>.>>>>
  </eqnarray*>
</body>

<initial|<\collection>
</collection>>