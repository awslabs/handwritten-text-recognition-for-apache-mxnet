/*
 * ASCLITE
 * Author: Jerome Ajot, Jon Fiscus, Nicolas Radde, Chris Laprun
 *
 * This software was developed at the National Institute of Standards and Technology by 
 * employees of the Federal Government in the course of their official duties. Pursuant
 * to title 17 Section 105 of the United States Code this software is not subject to
 * copyright protection and is in the public domain. ASCLITE is an experimental system.
 * NIST assumes no responsibility whatsoever for its use by other parties, and makes no
 * guarantees, expressed or implied, about its quality, reliability, or any other
 * characteristic. We would appreciate acknowledgement if the software is used.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS."  With regard to this software, NIST MAKES NO EXPRESS
 * OR IMPLIED WARRANTY AS TO ANY MATTER WHATSOEVER, INCLUDING MERCHANTABILITY,
 * OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "alignment_test.h"
#include "graphalignedsegment.h"
#include "alignedspeech.h"

AlignmentTest::AlignmentTest() {
	bench = new StdBenchmark();
}

AlignmentTest::~AlignmentTest()
{
	//delete bench;
}

void AlignmentTest::TestAll() {
	cout << "Testing insertions..." << endl;
	TestInsertions();
	cout << "OK!" << endl;
	
	cout << "Testing deletions..." << endl;
	TestDeletions();
	cout << "OK!" << endl;
}

void AlignmentTest::TestInsertions() {
	Alignment* ali;
	SegmentsGroup* sg;
	int testIndex = 2; // simple insertion, one segment	
	ali = GetAlignmentFor(testIndex, &sg);
	
	Segment* ref = sg->GetReference(0)[0];
	Segment* hyp = sg->GetHypothesis(0)[0];
	vector< Token* > refs = ref->ToTopologicalOrderedStruct();
	vector< Token* > hyps = hyp->ToTopologicalOrderedStruct();
	
	cout << "Single Insertion tests:" << endl;
	cout << "\tCheck speech and segment...";
	cout.flush();
	Speech* refSpeech = ref->GetParentSpeech();
	AlignedSpeech* asp = ali->GetOrCreateAlignedSpeechFor(refSpeech, false); 
	assert(refSpeech == asp->GetReferenceSpeech());
	AlignedSegment* asg = asp->GetOrCreateAlignedSegmentFor(ref, false);
	assert(ref == asg->GetReferenceSegment());
	assert(asg->GetTokenAlignmentCount() == 3);
	cout << " OK." << endl;
	
	cout << "\tCheck tokens...";
	cout.flush();
	TokenAlignment* ta = asg->GetTokenAlignmentAt(0);
	assert(ta->GetReferenceToken() == refs[0]);
	assert(ta->GetTokenFor("hyp") == hyps[0]);
	ta = asg->GetTokenAlignmentAt(1);
	assert(ta->GetReferenceToken() == NULL);
	assert(ta->GetTokenFor("hyp") == hyps[1]);
	ta = asg->GetTokenAlignmentAt(2);
	assert(ta->GetReferenceToken() == refs[1]);
	assert(ta->GetTokenFor("hyp") == hyps[2]);
	cout << " OK." << endl;
			
	testIndex = 4; // only insertion, one segment
	ali = GetAlignmentFor(testIndex, &sg);
	
	ref = sg->GetReference(0)[0];
	hyp = sg->GetHypothesis(0)[0];
	refs = ref->ToTopologicalOrderedStruct();
	hyps = hyp->ToTopologicalOrderedStruct();
	
	cout << "Only Insertion tests:" << endl;
	cout << "\tCheck speech and segment...";
	cout.flush();
	refSpeech = ref->GetParentSpeech();
	asp = ali->GetOrCreateAlignedSpeechFor(refSpeech, false); 
	assert(refSpeech == asp->GetReferenceSpeech());
	asg = asp->GetOrCreateAlignedSegmentFor(ref, false);
	assert(ref == asg->GetReferenceSegment());
	assert(asg->GetTokenAlignmentCount() == 3);
	cout << " OK." << endl;
	
	cout << "\tCheck tokens...";
	cout.flush();
	ta = asg->GetTokenAlignmentAt(0);
	assert(ta->GetReferenceToken() == NULL);
	assert(ta->GetTokenFor("hyp") == hyps[0]);
	ta = asg->GetTokenAlignmentAt(1);
	assert(ta->GetReferenceToken() == NULL);
	assert(ta->GetTokenFor("hyp") == hyps[1]);
	ta = asg->GetTokenAlignmentAt(2);
	assert(ta->GetReferenceToken() == NULL);
	assert(ta->GetTokenFor("hyp") == hyps[2]);
	cout << " OK." << endl;
	
	delete ali;
	delete sg;
}

void AlignmentTest::TestDeletions() {
	Alignment* ali;
	SegmentsGroup* sg;
	int testIndex = 3; // simple deletion, one segment	
	ali = GetAlignmentFor(testIndex, &sg);
	
	Segment* ref = sg->GetReference(0)[0];
	Segment* hyp = sg->GetHypothesis(0)[0];
	vector< Token* > refs = ref->ToTopologicalOrderedStruct();
	vector< Token* > hyps = hyp->ToTopologicalOrderedStruct();
	
	cout << "Single Deletion tests:" << endl;
	cout << "\tCheck speech and segment...";
	cout.flush();
	Speech* refSpeech = ref->GetParentSpeech();
	AlignedSpeech* asp = ali->GetOrCreateAlignedSpeechFor(refSpeech, false); 
	assert(refSpeech == asp->GetReferenceSpeech());
	AlignedSegment* asg = asp->GetOrCreateAlignedSegmentFor(ref, false);
	assert(ref == asg->GetReferenceSegment());
	assert(asg->GetTokenAlignmentCount() == 3);
	cout << " OK." << endl;
	
	cout << "\tCheck tokens...";
	cout.flush();
	TokenAlignment* ta = asg->GetTokenAlignmentAt(0);
	assert(ta->GetReferenceToken() == refs[0]);
	assert(ta->GetTokenFor("hyp") == hyps[0]);
	ta = asg->GetTokenAlignmentAt(1);
	assert(ta->GetReferenceToken() == refs[1]);
	assert(ta->GetTokenFor("hyp") == NULL);
	ta = asg->GetTokenAlignmentAt(2);
	assert(ta->GetReferenceToken() == refs[2]);
	assert(ta->GetTokenFor("hyp") == hyps[1]);
	cout << " OK." << endl;
		
	testIndex = 5; // only deletions, one segment
	ali = GetAlignmentFor(testIndex, &sg);
	
	ref = sg->GetReference(0)[0];
	hyp = sg->GetHypothesis(0)[0];
	refs = ref->ToTopologicalOrderedStruct();
	hyps = hyp->ToTopologicalOrderedStruct();
	
	cout << "Only Deletions tests:" << endl;
	cout << "\tCheck speech and segment...";
	cout.flush();
	refSpeech = ref->GetParentSpeech();
	asp = ali->GetOrCreateAlignedSpeechFor(refSpeech, false); 
	assert(refSpeech == asp->GetReferenceSpeech());
	asg = asp->GetOrCreateAlignedSegmentFor(ref, false);
	assert(ref == asg->GetReferenceSegment());
	assert(asg->GetTokenAlignmentCount() == 3);
	cout << " OK." << endl;
	
	cout << "\tCheck tokens...";
	cout.flush();
	ta = asg->GetTokenAlignmentAt(0);
	assert(ta->GetReferenceToken() == refs[0]);
	assert(ta->GetTokenFor("hyp") == NULL);
	ta = asg->GetTokenAlignmentAt(1);
	assert(ta->GetReferenceToken() == refs[1]);
	assert(ta->GetTokenFor("hyp") == NULL);
	ta = asg->GetTokenAlignmentAt(2);
	assert(ta->GetReferenceToken() == refs[2]);
	assert(ta->GetTokenFor("hyp") == NULL);
	cout << " OK." << endl;
	
	delete ali;
	delete sg;
}

Alignment* AlignmentTest::GetAlignmentFor(int testIndex, SegmentsGroup** sg) {
	Alignment* ali = NULL;
	GraphAlignedSegment* gas = bench->GetResult(testIndex);
 	*sg = bench->GetTest(testIndex);
	ali = new Alignment();
	ali->AddSystem("", "hyp");
	ali->AddGraphAlignedSegment(gas, "hyp", *sg);
	return ali;
}
