#ifndef _PARALLEL_H_
#define _PARALLEL_H_

#include <iostream>
#include <thread>
#include <queue>
#include <future>
//#include "my_datastructures.h"

namespace concurrent{

	template <class Func,typename job>
	class PARALLEL{
		private:

		class Project{
		private:
			
			std::mutex next_mutex;
			std::queue<job> *whole_batch;

		public:
			

			Project():whole_batch(nullptr){}
			Project(std::queue<job>& batch):whole_batch(&batch){
			}
			job* next_job(){
				std::lock_guard<std::mutex> lockGuard(next_mutex);
				if(whole_batch == nullptr){throw "unavailable batch exception";}
				if(whole_batch->size() > 0){

					job* next = new job;
					*next =whole_batch->front();
					whole_batch->pop();
					return next;
				}else{
					return nullptr;
				}
			}
			
		};
		
		class Resolver{
			public:
			

			void Master(Project* workload,bool* over,const Func* func){
				
				job* next;
				do{
					next = workload->next_job();
					while(next != nullptr){
		
							try{
									(*func)(*next);	
									delete next;
								}catch(const char* c){
									std::cout<<c;
									throw;
								}catch (std::bad_alloc& ba)
								{
									std::cerr << "bad_alloc caught: " << ba.what() << '\n';
									throw;
								}catch(...){
									std::cerr << "unknown excption";
									throw;
								}
						next = workload->next_job();
					}
				}while(!(*over));
			}
		};
		
		Project * workload;
		Func* func;
		int NUM_THREADS;
		public:	
		
		PARALLEL(const Func& func,std::queue<job>& jobs,int NUM_THREADS):NUM_THREADS(NUM_THREADS){
			
			if(NUM_THREADS <= 0){throw "unsupported thread number";}
			workload=new Project(jobs);
			this->func=new Func(func);
		}

		~PARALLEL(){
			delete workload;
			delete func;
		}

		void pstart(bool& over){

			std::thread master[NUM_THREADS];
			Resolver resolve;

			for (int i = 0; i < NUM_THREADS; ++i){
				master[i] =  std::thread(&Resolver::Master,&resolve,workload,&over,func);
			}

			for (int i = 0; i < NUM_THREADS; ++i)
			{
				master[i].join();
			}

		}

		void pstart(){

			bool read=true;
			std::thread master[NUM_THREADS];
			Resolver resolve;

			for (int i = 0; i < NUM_THREADS; ++i){
			    	master[i] =  std::thread(&Resolver::Master,&resolve,workload,&read,func);
			}

			for (int i = 0; i < NUM_THREADS; ++i)
			{
				master[i].join();
			}

		}				
	};
	template<class T>
	class Output{
	private:
		std::queue<T>* out;
		std::mutex shared_mutex;
	public:

		Output(){

			out = new std::queue<T>;

		}

		void push(const T& element){
			
			std::lock_guard<std::mutex> lockGuard(shared_mutex);
			try{
				out->push(element);
			}catch(const char* exc){
				std::cout<<exc;
				throw;
			}

		}

 		T pop(){
			
			std::lock_guard<std::mutex> lockGuard(shared_mutex);
			try{
				T temp;
				temp = out->front();
				out->pop();
				return temp;
			}catch(const char* exc){
				std::cout<<exc;
				throw;
			}

		}

		unsigned size(){
			std::lock_guard<std::mutex> lockGuard(shared_mutex);
			return out->size();
		}


		~Output(){
			delete out;	
		}
	};
}
#endif


