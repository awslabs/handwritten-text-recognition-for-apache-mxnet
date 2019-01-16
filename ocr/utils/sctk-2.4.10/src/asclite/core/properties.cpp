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
 
/**
 * Contain the global properties of the aligner.
 * This object in created once and for all during the argument parsing.
 */

#include "properties.h" // class's header file

/**
 * Internal representation of the properties
 */
map<string, string> Properties::properties;
bool Properties::dirty = true;

/**
 * Set the property to the desire value
 */
void Properties::SetProperty(const string& name, const string& value)
{
    properties[name] = value;
	dirty = true;
}

/**
 * Set the properties with a all set of predefined values
 */
void Properties::SetProperties(const map<string, string>& props)
{
    dirty = true;
    properties = props;
	
	if(properties.find("align.case_sensitive") == properties.end())
	{
		properties["align.case_sensitive"] = "false";
	}
	
	if(properties.find("align.fragment_are_correct") == properties.end())
	{
		properties["align.fragment_are_correct"] = "false";
	}
	
	if(properties.find("align.optionally") == properties.end())
	{
		properties["align.optionally"] = "none";
	}
	
	if(properties.find("align.timepruneoptimization") == properties.end())
	{
		properties["align.timepruneoptimization"] = "false";
	}
	
	if(properties.find("align.timepruneoptimizationthreshold") == properties.end())
	{
		properties["align.timepruneoptimizationthreshold"] = "0";
	}
	
	if(properties.find("align.timewordoptimization") == properties.end())
	{
		properties["align.timewordoptimization"] = "false";
	}
	
	if(properties.find("align.timewordoptimizationthreshold") == properties.end())
	{
		properties["align.timewordoptimizationthreshold"] = "0";
	}
	
    if(properties.find("align.speakeroptimization") == properties.end())
	{
		properties["align.speakeroptimization"] = "false";
	}
	
	if(properties.find("align.adaptivecost") == properties.end())
	{
		properties["align.adaptivecost"] = "false";
	}
    
	if(properties.find("recording.maxspeakeroverlaping") == properties.end())
	{
		properties["recording.maxspeakeroverlaping"] = "4";
	}
	
	if(properties.find("recording.maxoverlapinghypothesis") == properties.end())
	{
		properties["recording.maxoverlapinghypothesis"] = "1";
	}
	
	if(properties.find("recording.maxnbofgb") == properties.end())
	{
		properties["recording.maxnbofgb"] = "2";
	}
	
	if(properties.find("recording.nbrdifficultygb") == properties.end())
	{
		properties["recording.nbrdifficultygb"] = "16";
	}
	
	if(properties.find("recording.minnbrdifficultygb") == properties.end())
	{
		properties["recording.minnbrdifficultygb"] = "0";
	}
	
	if(properties.find("align.memorycompressionblock") == properties.end())
	{
		properties["align.memorycompressionblock"] = "64";
	}
}

/**
 * Retrieve the value of the specified property
 */
string Properties::GetProperty(const string& name)
{
	dirty = false;
    return properties[name];
}

/**
 * Initialize the properties.
 * Nothing to do for now there...
 */
void Properties::Initialize() 
{
	map<string, string> props;
	props.clear();
	SetProperties(props);
}
