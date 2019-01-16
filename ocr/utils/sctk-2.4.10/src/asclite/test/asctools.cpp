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

#ifdef WIN32
//#include <windows.h>
#endif

#include "asctools.h"

#ifdef __APPLE__
#include <mach/mach.h>
#endif

/** Get the current time in ms */
double timerStart()
{
  timeval tv;
  gettimeofday(&tv, NULL);
  return((double)(tv.tv_sec*1000000+tv.tv_usec)/1000);
}

/** Return the difference between 2 times ms */
double timerEnd(double start)
{
	timeval tv;
  gettimeofday(&tv, NULL);
  return(((double)(tv.tv_sec*1000000+tv.tv_usec)/1000) - start);
}


/** Print on the screen the memory usage */
int MemoryUsage()
{
#ifdef WIN32
	//return (int) getMemoryUsed();
	
	//MEMORY_BASIC_INFORMATION mbi;
	//DWORD      dwMemUsed = 0;
	//PVOID      pvAddress = 0;
	
	//memset(&mbi, 0, sizeof(MEMORY_BASIC_INFORMATION));
	
	//while(VirtualQuery(pvAddress, &mbi, sizeof(MEMORY_BASIC_INFORMATION)) == sizeof(MEMORY_BASIC_INFORMATION))
	//{
	//	if(mbi.State == MEM_COMMIT && mbi.Type == MEM_PRIVATE)
	//		dwMemUsed += mbi.RegionSize;
		
	//	pvAddress = ((BYTE*)mbi.BaseAddress) + mbi.RegionSize;
	//}
	
	//return (int) dwMemUsed;
	return 0;
#else
#ifdef __APPLE__
	kern_return_t			err;
	mach_msg_type_number_t	count;
	task_basic_info_data_t	taskinfo;
	
	count = TASK_BASIC_INFO_COUNT;
	err = task_info( mach_task_self(), TASK_BASIC_INFO, (task_info_t)&taskinfo, &count );
	if (err != KERN_SUCCESS) return 0;
	
	return (int) taskinfo.resident_size / 1024;	
#else //*NUX	
	char buffer[256];
  ifstream statusfile("/proc/self/status");
	
	if (!statusfile.is_open())
  {
		cout << "Error opening file" << endl;
		return 0; 
	}
	
  while (!statusfile.eof())
  {
    statusfile.getline(buffer,100);
		
		if((buffer[4] == 'z') && (buffer[0] == 'V') )
		{
			char* pch;
  		pch = strtok(buffer," ");
			pch = strtok(NULL, " ");
			
			return std::atoi(pch);
		}
  }
	
	return 0;
#endif //__APPLE__
#endif //WIN32
}
