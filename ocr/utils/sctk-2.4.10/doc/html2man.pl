#!/usr/local/bin/perl

sub basename {
  $slash = rindex($_[0], "/");
  $suffix = rindex($_[0], $_[1]);
  if ($suffix == -1) {$suffix=length($_[0])};
  return substr($_[0], $slash+1, $suffix-$slash-1);
}

sub dirname {
  $slash = rindex($_[0], "/");
  if ($slash == -1) {return "."};

  return substr($_[0], 0, $slash);
}

sub DoHtml {

  $preformat = 0;
  $page = &basename($_[0], ".htm");
  $dir = &dirname($_[0]);
  $sect = chop($dir);
  if ($sect == "n") {
    $sect = chop($dir) . $sect;
  };
  $sect = "1";
  $dir = &dirname($dir);
  $infile = $dir . "/" . $page . ".htm";
  $outfile = $dir . "/" . $page . "." . $sect;
  print "\n*** $dir $sect $page *** \n";
  open(INP,$infile) || die "Can't open '$infile' for reading: $!";
  if (! &GetTitle($page,$sect)) {  
    print "$infile: Did not find a valid <TITLE> line\n";
    close(INP);
    return;
  }
  open(OUT,"> $outfile") || die "Can't open '$outfile' for writing: $!";
  print "Converting: $infile to $outfile\n";
  &PrintHeader($page,$sect,$TITLE[0]);

  while (<INP>) {
    $result = &DoLine($_);
    if ($result == 2) {
      close(INP);
      close(OUT);
      return;
    }
    if (&DoLine($_) == 1) {
	s#&lt#<#g;
	s#&gt#>#g; 
       print OUT;
    }
  }
  close(INP);
  close(OUT);
  return;
}

sub GetTitle {
  while (<INP>) {
    if (m#<TITLE>.*MAN page for:#i) {
      s#<TITLE>##i;
      s#MAN page for:##i;
      s#</TITLE>##i;
      @TITLE = split;
      return 1;
    }
    if (m#<TITLE>#) {
      return 1;
    }
  }
  return 0;
}

sub PrintHeader {
      print OUT ".TH $_[0] $_[1] \"\" \"\" \"\" \"$_[2]\"\n";
##      print OUT ".so /usr/share/lib/tmac/sml\n";
##      print OUT ".so /usr/share/lib/tmac/rsml\n";
}

sub DoStrip {
  if (m#<.*TITLE>#i) { return 1};
  if (m/<.*HEADER>/i) { return 1};
  if (m/<.*HEAD>/i) { return 1};
  if (m/<.*BODY>/i) { return 1};
  if (m/<.*HEADER>/i) { return 1};
  if (m/<.*BLINK>/i) { return 1};
  if (m/^<A HREF="#toc/i) {return 1};
  if (s#<A HREF=".*">#\\*L#i) {
      s#</A>#\\*O#i;
  }
  return 0;
}

sub DoLine {

  if (m#^ *</PRE>#i) {
    s#^ *</PRE>#.DE\n\\*O#i;
    $preformat = 0;
    return 1;
  }
###  if ($preformat) {return 1};
  if (m#^ *<PRE>#i) {
    s#^ *<PRE>#\\*C\n.DS#i;
    $preformat = 1;
    return 1;
  }
  s#<IMG SRC=.*>##i;
  if (m#<A NAME="toc">#) {return 2};
  s#<A NAME=".*">##i;
  s#<\!\-\-#.\\\" #;
  s#\-\->##;
  if (m#^<HR.*>$#i) {return 0};
  s#<HR>##i;
  s#<HR .*>##i;
  if (&DoStrip($_)) {return 0}
  &DoFont($_);
  if (! &DoList($_)) {return 0};
  if (! &DoSection($_)) {return 0}
  &DoEscape($_);
  &DoPara($_);
  if (! $preformat) {
    if (m/^$/) {return 0};
    s#^[ \t]*##;    
    s#<.*>##g;    
  }
  return 1;
}

sub DoEscape {
  s#&gt;#>#i;
  s#&lt;#<#i;
}

sub DoSection {
  if (m#^ *Table of Contents *$#) {return 0};
  if (m#^<A NAME=".*">$#i) {return 0;}
  s#<A NAME=".*">##i;
  s#^.*<H2>#.SH #i;
  s#^.*<H3>#.SH #i;
  if (m#^</.*>$#i) {return 0};
  s#</.*>##ig;
  s#^<BR>$#.br#i;
  s#<BR>#\n.br\n#i;
  return 1;
}


#================
# Paragraph breaks: <P>
#
sub DoPara {
  s/^<P>/.PP\n/;
  s/<P>/\n.PP\n/g;
  return;
}

#================
# Lists
#    <DL><DT><DD>
#
sub DoList {
  if (m#^ *</DT> *$#i) {return 0};
  if (m#<DT> *$#i) {chop};
  s#^ *<DT>#.LI \"#i;
  s#<DT>#\n.LI \"#i;
  s#</DT>#\"#i;

  if (m#^ *<DD> *$#i) {return 0};
  if (m#^ *</DD> *$#i) {return 0};
  if (m#^</LI>$#i) {return 0};
  s#<DD>##i;
  s#</DD>##i;

  s#^.*<DL> *$#.VL 4m#i;
  s#<DL>#\n.VL 4m\n#i;

  s#^.*</DL> *$#.LE#i;
  s#<DL>#\n.LE\n#i;

  s#^.*<UL> *$#.RS#i;
  s#<UL>#\n.RS\n#i;

  s#^.*</UL> *$#.RE#i;
  s#</UL>#\n.RE\n#i;

  s#^.*<OL> *$#.AL#i;
  s#<OL>#\n.AL\n#i;

  s#^.*</OL> *$#.LE#i;
  s#</OL>#\n.LE\n#i;

  s#</LI>##i;
  s#^<LI>$#.LI#i;
  s#^ *<LI> *$#.LI#i;
  s#<LI>#.LI\n#i;

  return 1;
}

#================
# Font transitions: 
#   <B>  => \*L
#   </B> => \*O
#   <I>  => \*W
#   </I> => \*O
#
sub DoFont {
  s# *</B> *_ *<B> *#_#ig;
  s#</B> *<B># #ig;
  s#<B>#\\*L#ig;
  s#</B>#\\*O#ig;

  s# *</I> *_ *<I> *#_#ig;
  s#</I> *<I># #ig;
  s#<I>#\\*W#ig;
  s#</I>#\\*O#ig;
  s/^ *<P>/<P>/;
  s/^<P>$/.PP/;
  s/<P>/.PP\n/g;
  s/<P>/.PP\n/g;
  return;
}

#================
# Process all HTM file names
for $arg (@ARGV) {
  $isTitle=0;
  &DoHtml($arg);
}

exit
