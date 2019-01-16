#!/bin/sh


for width in 2 3 4 5 6 ; do
  for f in tests.hyp tests.ref ; do
    cat $f | \
    perl -pe '
        use Encode;
        use encoding "utf8";  
        binmode STDIN, ":utf8";
        binmode STDOUT, ":utf8";
        my $width = '$width';
        my @offsets = (undef, under, 0x100, 0x900, 0x11111, 0x300000, 0x5000000);
        my $offset = $offsets[$width];
#        printf("offset = %x\n",$offset);
        @words = ("-offee", "along", "which",  "problem","private", "age-old",
                  "del_withnull", "skip", "substitution", "sub_withnull", "-icks", "(ten-)", "right",
                  "find", "fish", "ticks", "milk", "fig-",  "have", "flag", "flag", "long", "(c-)", 
                  "ten", "and", "tea", "del", "the", "fee", "him", "fi-", "th-", "had", "age", "old", "(a)", "(b)", "(c)", "(the)",
                  "(d)", "(e)", "can", "this",
                  "-lk", "-ik", "-him", "bc", "gh", "as", "cd", "th", "is",
                  "aa", "bd", "e-", "ef", "en", "uh", "of", "my", "go", "ih", "an", "fi",
                  "a", "b", "c", "d", "e", "f", "g", "h", "s", "t", "w", "i", "j", "k", "n", "y", "x"
                  );
        
        foreach my $w(@words){
          my $x = "";
          foreach my $char(split(//, $w)){
            if ($char =~ /([\(\)\-])/){
              $x .= $char;
            } else {
              ## print STDOUT "$w $char ".(ord($char)-ord('A'))." ".chr($offset +  ord($char) - ord('A'))."\n";
              $x .= chr($offset +  ord($char) - ord('A'));
            }
          }
          push @repls, $x;
        }

        if ($_ !~ /^;;/){
           for($w=0; $w < @words; $w++){
              $pat = quotemeta($words[$w]);
              $_ =~ s/^$pat/$repls[$w]/g;
              $_ =~ s/([{\/])$pat/$1$repls[$w]/g;
              $_ =~ s/ $pat/ $repls[$w]/g;
           }
        }' > $f.utf8-${width}bytes
  done
done
