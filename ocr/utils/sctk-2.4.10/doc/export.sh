#!/bin/sh

echo Making the revisions HTML file
echo '<!-- $Id: export.sh,v 1.2 2001/10/11 12:11:14 jon Exp $ -->
<HTML><HEAD>
<CENTER><TITLE>SCLITE revisions</TITLE>
</HEAD>
<BODY></CENTER><p><hr>

<H1> 
<A NAME="revisions_name_0">
<strong>
<A HREF="sclite.htm#sclite_name_0">Sclite</A> Revision.txt </A>
</strong>
</H1>
<p>
<pre>'  > revis.htm
cat ../revision.txt >> revis.htm
echo '</pre>
</body>
</html>' >> revis.htm

normalize(){
    sed -e 's/<a/<A/g' -e 's:</a:</A:g' \
    	-e 's/<pre/<PRE/g' -e 's:</pre:</PRE:g' \
	-e 's/<ul/<UL/g' -e 's:</ul:</UL:g' \
	-e 's/<hl/<HL/g' -e 's:</hl:</HL:g' \
	-e 's/<dl/<DL/g' -e 's:</dl:</DL:g' \
	-e 's/<dt/<DT/g' -e 's:</dt:</DT:g' \
	-e 's/<dd/<DD/g' -e 's:</dd:</DD:g' \
	-e 's/<strong/<STRONG/g' -e 's:</strong:</STRONG:g' \
	-e 's/<br/<BR/g' \
	-e 's/<hr/<HR/g' \
	-e 's/<p/<P/g' \
	-e 's/<A name=/<A name=/g' \
	-e 's/<A href=/<A href=/g' 
}

make_page(){
    name=$1
    htms="$2"
    echo Making the $name manual page
    rm -rf foo
    mkdir foo 
    for f in $htms ; do
	echo "   Normalizing working on $f"
	cat $f | normalize | cat >> foo/$name.htm
    done
    (cd foo ; ../html2man.pl $name) ;	
    cat foo/*.1 | \
	perl -pe 'if ($_ =~ /SYNOPSIS/) 
	{print "NOTE: This manual page was created automatically from\n".
	       "HTMl pages in the sclite/doc directory.  This manual page does not\n".
	       "include output file examples.  The author suggests using a HTML browser\n".
	       "for reading the sclite documentation.\n.PP\n"; s/\.sys\./".sys"./;}' > $name.1
    rm -rf foo
}

make_page sclite "sclite.htm options.htm infmts.htm"
make_page rover "rover.htm"
make_page sc_stats "sc_stats.htm st_opt.htm"



