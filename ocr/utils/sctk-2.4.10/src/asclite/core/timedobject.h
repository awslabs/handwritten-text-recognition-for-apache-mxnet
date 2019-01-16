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

#ifndef TIMEDOBJECT_H
#define TIMEDOBJECT_H

#include "stdinc.h"
#include "id.h"

/**
 * Represent a timed object.
 * A timed object is represented by a start time and a end time.
 */
class TimedObject
{
    public:		
        // class destructor
        virtual ~TimedObject() {}
            
        int GetStartTime() { return startTime; } // returns the value of startTime
        int GetDuration() { return endTime - startTime; } // returns the value of duration
        int GetEndTime() { return endTime; } // returns the value of endTime
        
        /** Checks if this TimedObject overlaps with the given TimedObject */
        bool OverlapWith(TimedObject* other);
        
        /** 
         * Returns the status of the time of this object.
         * True: You can use the time it's a real one.
         * False: The times are virtual or guessed.
         */
        bool IsTimeReal() { return (startTime >= 0 && endTime >= 0); }
        
        bool Equals(TimedObject* to);
		
		ulint GetsID() { return s_id; }
		
		int TimeSafeDivider();
        
    protected:
        /** Initializes this TimedObject based on a start time and duration. */
        void* InitWithDuration(const int& _startTime = -1, const int& _duration = -1);
        
        /** Checks that start time and duration are valid for this TimedObject. Extension point for subclasses. */
        virtual bool AreStartTimeAndDurationValid(const int& _startTime, const int& _duration);
        
        /** Initializes this TimedObject based on a start time and end time. */
        void* InitWithEndTime(const int& _startTime = -1, const int& _endTime = -1);
        
        /** Checks that start time and end time are valid for this TimedObject. Extension point for subclasses. */
        virtual bool AreStartTimeAndEndTimeValid(const int& _startTime, const int& _endTime);
        
        /** Returns a string representation of this TimedObject. */
        virtual string ToString();
        
        // class constructor
        TimedObject() { s_id = ID::GetID(); }
            
    private:
        /**
         * The start time of the token.
         */
        int startTime;
		
		ulint s_id;
        
    protected:
        /**
         * The end time of this token.
         */
        int endTime;
};

#endif // TIMEDOBJECT_H
