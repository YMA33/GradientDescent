#ifndef _SWAP_H_
#define _SWAP_H_

/**
	This header contains macro definitions to streamline the swapping of data.
	The main macro defined is SWAP(a,b) that swaps the content of a and b as
	long as they are of the same type and they support the = operator.

	Notice that the type of the a and b is not passed and it is infered using typeof()
	operator.
*/

//#define SWAP(a,b) {typeof(a) tmp=a; a=b; b=tmp;}
#define SWAP(a,b) {auto tmp=a; a=b; b=tmp;}

/* macro to swap content of STL-like containers/maps */
#define STL_SWAP(a,b) a.swap(b);
/* datapath swaps */
#define DP_SWAP(a,b) a.Swap(b);

/* macro to do copy swap.

	 Warning: this is dangerous. Use only if basic datatypes are stored
in the class. DO NOT USE IF CONTAINERS ARE STORED INSIDE. Instead break the swap into mini-swaps using the above two macros

*/
#define MEM_SWAP(a,b)							 \
	{ \
		typedef typeof(a) obj_type;									\
		char storage[sizeof (obj_type)];							\
		memmove (storage, &(a), sizeof (obj_type));		\
		memmove (&(a), &(b), sizeof (obj_type));				\
		memmove (&(b), storage, sizeof (obj_type));			\
	}

/** macro to define the swap. If a class is safe to swap using
		memmove, just call this macro in the public declaration part */

#define MEMCOPY_SWAP()													\
	void Swap(className& other) {						\
		MEM_SWAP((*this), other);										\
	}

/** macros for declaring swap functions */

#define SWAP_DECLARATION() void Swap(className& who)
#define SWAP_DEFINITION(who) void Swap(className& who)
#define SWAP_DEFINITION_C(classN, who) void classN::Swap(classN & who)


#endif //_SWAP_H_
