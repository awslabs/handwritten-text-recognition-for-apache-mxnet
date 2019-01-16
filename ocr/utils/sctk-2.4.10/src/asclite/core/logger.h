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
 * Logging methods
 */
 
#ifndef LOGGER_H
#define LOGGER_H

#include "stdinc.h"

/** 
 * Logging methods
 */
class Logger
{
    public:
        /** class destructor */
        ~Logger() {}
        
        /** 
         * Set the logger to log in the path specified as an argument 
         */
        static void logInFile(const string& file);

        /** 
         * Log on stdout
         */
        static void logOnStdout();

        /** 
         * Change log level
         */
        static void setLogLevel(const int& level);

        /**
         * Return the stream to output the loggin
         * level must be a int between 0 and 5
         * where :
         *   0 ===> Silence 
         *   1 ===> Fatal
         *   2 ===> Error 
         *   3 ===> Warn
         *   4 ===> Info (default)
         *   5 ===> Debug
		 *   6 ===> Alignment
		 *   7 ===> Alignment only
         */ 
        void log(const int& level, const string& message);
        void log(const int& level, const char* const message);
		
		int getLogLevel() { return log_level; }
		
		bool isAlignLogON() { return ( (log_level == 6) || (log_level == 7) );  }
        
        string LevelToString(const int& level);
        
        static Logger* getLogger();
    
    protected:
        /** class constructor */
        Logger() {}
    
    private:
        static Logger* rootLogger;
        static ostream* where;
        static int log_level;
};

/* ---------------------------------- */
/* Macro definition to use the logger */
/* ---------------------------------- */

#ifndef LOG_ALIGN
#define LOG_ALIGN(_l,_m) _l->log(6, (_m));
#endif

#ifndef LOG_DEBUG
#define LOG_DEBUG(_l,_m) _l->log(5, (_m));
#endif

#ifndef LOG_INFO
#define LOG_INFO(_l,_m) _l->log(4, (_m));
#endif

#ifndef LOG_WARN
#define LOG_WARN(_l,_m) _l->log(3, (_m));
#endif

#ifndef LOG_ERR
#define LOG_ERR(_l,_m) _l->log(2, (_m));
#endif

#ifndef LOG_FATAL
#define LOG_FATAL(_l,_m) _l->log(1, (_m));
#endif

/****************************************
 * Output Errors                        *
 ****************************************/

const int E_OK       = 0;
const int E_LOAD     = 1;
const int E_COND     = 2;
const int E_FILTER   = 3;
const int E_ARGS     = 4;
const int E_MISSINFO = 5;
const int E_INVALID  = 6;
const int E_LZO      = 7;

#endif
