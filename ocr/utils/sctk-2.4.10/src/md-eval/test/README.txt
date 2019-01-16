Each test is described by comments in the .ref.rttm file.

STRUCTURAL METADATA   Tests to be run with options -w and -W
---------------------
md_test1  There should be no errors: three speakers (of different types),
          four SU's, and instances of all scoreable lexeme subtypes.

md_test2  Has an SU starting at  7.0 that should be completely non-scored
          has an SU starting at 10.0 that should be scored and without error
          has an SU starting at 13.0: type incomplete in ref and question in sys
          has an SU starting at 16.0 that should be scored and without error
          has an SU starting at 19.0 that in the ref ends just after the end
                                  of scored time (sys is entirely within
                                  scored time) -- illustrating some program
                                  logic for an earlier scoring rule.
        
          In the output:
             The SU starting at 7.0 should not count because it ends before
                scored time begins.
             The SU starting at 13.0 should have a type mismatch.
                 and should be an SU substitution error "in terms of # of SUs".
             There should be four scored SU's.
     
md_test3   Like test2, but the SU starting at 19.0 ends exactly at the end
           of scored time.  Contrasts with the md_test2 SU starting at 19.0.

md_test4   Has some oddball but syntactically legal speaker names.
           The program handles them OK.  SYSTEMS SHOULD NOT EMULATE THESE NAMES!!!
           Md_test4 also tests that Nref in terms of # of SU's is the same as
           Nref for exact SU boundaries in terms of words. 

md_test5   Tests subtype other.  Exercises some dark corners of the RTTM spec.

md_test6   This resembles tests 1 - 4
           The SU starting at 7.0 ends exactly at the start of
           scored time.  Thus, a system processing this SU would look at exactly
           zero point zero seconds of audio data.  The SU starting at 7.0 should
           not get scored.

md_test7   Has SU's with systematically varying number of words of overlap
           between ref SU and sys SU.  But all the words are "blah".
           This will often reveal changes to the word alignment algorithm
           (see comment at top of md_test7.ref.rttm for expected output)
 
           See md_test24 for a related variation. 

md_test8   Has EDIT of subtype REVISION, with explicit_editing_term.
           Has EDIT with edit subtype mismatch and filler subtype mismatch.
      
md_test9   Has ref=sys match on a filler and a revision.  Output should be "no errors".

           Because the words are hyphenated, each counts as two words.

md_test10  Has fillers and edits with time mismatches.  The word-warping should
           be apparent in the output, and it is.

           Because the words are hyphenated, each counts as two words.

md_test11  See the comment at the top of the ref file.  Output OK.
           (It may be useful to check this with time-based command-line options
            as well as word-based.)

           Because the words are hyphenated, each counts as two words.

md_test12  See the comment at the top of the ref file.

md_test13  The SU starting at 16.0 is out of order, but md-eval should sort the lines.
           The sys.rttm file has *negative* durations for the SEGMENT, SPEAKER, and SU
           records that start at time 10.0    Will md-eval do something appropriate???

md_test14  This has the ref and sys that will be used in md_test15  md_test16 and
           md_test17, but here (in md_test14) the UEM file has the whole thing as
           eval'd time

           Because the words are all hyphenated, each counts as two words.

md_test15 and Test16
           The UEM file has some of the filler and edit words in no-eval'd time.

           Because the words are all hyphenated, each counts as two words.

md_test17  Here, the UEM file has the entire FILLER and EDIT in no-eval'd time.

           Because the words are all hyphenated, each counts as two words.

md_test18  The ref carves up the time interval from 16.0 to 28.0 into three four-second SU's.
           The sys carves up that time interval differently.  Thus, the ref and sys carve up
           the time interval from 16.0 through 28.0 into different numbers of SU's, with
           different boundaries. 

           The expected scoring output is:
           ;;    SU (exact) end detection statistics -- in terms of reference words
           ;;                Nref  Ndel  Nins  Nsub  Nerr     %Del   %Ins   %Sub   %Err
           ;;         ALL       5     2     4  <ns>     6    40.00  80.00   <ns> 120.00
           ;;
           ;;    SU detection statistics -- in terms of # of SUs
           ;;                Nref  Ndel  Nins  Nsub  Nerr     %Del   %Ins   %Sub   %Err
           ;;         ALL       5     0     2     0     2     0.00  40.00   0.00  40.00

md_test19  Vaguely the same idea as test18.  The alignment of SU's is ambiguous, but
           the legal SU alignment giving minimum SU detection error will give
           ;;    SU (exact) end detection statistics -- in terms of reference words
           ;;                  Nref  Ndel  Nins  Nsub  Nerr     %Del   %Ins   %Sub   %Err
           ;;           ALL       5     2     3  <ns>     5    40.00  60.00   <ns> 100.00
           ;;
           ;;    SU detection statistics -- in terms of # of SUs
           ;;                  Nref  Ndel  Nins  Nsub  Nerr     %Del   %Ins   %Sub   %Err
           ;;           ALL       5     0     1     0     5    00.00  20.00   0.00  20.00

           However, with a different SU alignment, the output could be
           ;;    SU (exact) end detection statistics -- in terms of reference words
           ;;                  Nref  Ndel  Nins  Nsub  Nerr     %Del   %Ins   %Sub   %Err
           ;;           ALL       5     2     3  <ns>     5    40.00  60.00   <ns> 100.00
           ;;
           ;;    SU detection statistics -- in terms of # of SUs
           ;;                  Nref  Ndel  Nins  Nsub  Nerr     %Del   %Ins   %Sub   %Err
           ;;           ALL       5     2     3     0     5    40.00  60.00   0.00 100.00
           

md_test20  Like test1, except that there is an SU from 7.0 to 10.0, which ends exactly
           at the beginning of scored time: that SU should not count in the scoring.

           There should be 4 scored SU's with no errors.

md_test21  The ref and sys files are like test1, but the UEM file differs.
           For the SU from 13.0 to 16.0, only the time from 13.0 to 14.0 is eval'd time,
           ... which means its midpoint is non-eval'd, which means that SU should not count.
           For the  SU from 19.0 to 21.0, the time from 19.0 to 20.9 is UEM-scored time.
           There should be 3 scored SU's with no errors.

md_test22a Has 11.0 through 13.0 covered by a NOSCORE record.

           Has a SPEAKER record starting at 14.0 and an SU starting at 15.0 that have
           their midpoints during a NON-LEX.  Both should count in the scoring.

           Has an SU starting at 20.0 whose midpoint is after the end of UEM eval'd time.
           It should not count.

md_test23  Has 11.0 through 13.0 no-eval'd via the UEM file.
           Compare with md_test22a

md_test24  Has SU's with systematically varying number of words of overlap
           between ref SU and sys SU.  Unlike md_test7,  the words are distinct
           enough that ref and sys words with the same times should align in
           all SU's except the one from 12.0 through 20.0

           The SU from 12.0 through 20.0 tests the linear interpolation in the word-time warping.
           If you run it with options to generate the detailed output, you will see the following.

              Both ref and sys have a word from time 9.0 to 10.0, which get aligned
              to each other, so ref vs. sys times are identical at time 10.0

              The next word that gets matched up is from 14.0 to 15.0 in the ref
              and from 13.0 to 14.0 in the sys.  So sys time 13.0 is warped to
              the corresponding ref time (14.0)

              In between those matched-up words, there lie 4.0 seconds in the ref
              and 3.0 seconds in the sys, and (via straight line linear interpolation)
              each 1.0 of those three sys seconds equals 1.33 ref seconds.  The
              last of those three sys seconds has an inserted word (from sys time
              12.0 to 13.0).  The end time of that inserted sys word will be warped
              to 14.0 (see the end of the preceding paragraph).  So, the beginning of
              that word should be warped to 1.33 seconds earlier, which is 12.67

md_test25  This is an excerpt that blew up a previous version of md-eval
             (Sue Tranter provided this example)
              
md_test26  This has a region that is excluded by the UEM file and is NOSCORE in the RTTM
md_test27  This is md_test26 without the NOSCORE in the RTTM
md_test28  This is md_test26 without the exclusion in the UEM

md_test29  This is sd_test6 to show its output with the structural MDE command line



Speaker Diarization     Tests to be run with option -1 but neither -w nor -W
---------------------
sd_test1  spkr2 is in no-eval'd time and spkr4 is in no-eval'd time.
          There should be three scored speakers: two adult_females and one child
          with no misses and no false-alarms.

sd_test2  tests extension of no-score zone surrounding non-lex token
          see the details in the comments in the ref.rttm file
          sd_test2 and sd_test3 look at opposite directions

sd_test3  tests extension of no-score zone surrounding non-lex token
          see the details in the comments in the ref.rttm file
          sd_test2 and sd_test3 look at opposite directions

sd_test4  tests the error that was fixed in md-eval-v15.pl
          In the output of this test, the amount of MISSED SPEECH and
          of MISSED SPEAKER TIME should be 0.21 seconds.  There should
          be one missed word.   The amount of FALARM SPEECH and of
          FALARM SPEAKER TIME should be 26.73 seconds.
          
sd_test5  <distribution suppressed>

sd_test6  shows behavior when a system speaker pathologically overlaps with itself
          (this relates to the changes introduced in md-eval-v17)
          Note: this is duplicated as md_test29

