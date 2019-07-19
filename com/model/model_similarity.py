# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein
数据清洗和生成2个新表
1\职位清单
2\备选职位说明分词表
"""
import sys
from gensim import corpora, models, similarities
from collections import defaultdict
import statistics as stat
import time

from com.hr.base.public import * 
from com.hr.object.obj_resource import * 
from com.hr.object.obj_group import * 

class sim_tfidf(baseObject):
    # tfidf 模型计算
    def __init__(self,items,pic=0.02,process=0):
        self.source_seg = items
        self.colName ="tfidf_similarity"         
        self.title = items[0]["title"]
        self.classes = items[0]["class"]
        self.trade = items[0]["trade"]        
        self.max_ave = 0
        self.max_index = 0 
        self.max_pid = []
        self.old_textList = []
        self.old_weightList = []
        self.dictionary =self.__getDict(items)
        self.new_corpus =self.__getNewCorp()
        self.__calcSim(pic,process)
              
    def __getDict(self,items):
        rates=defaultdict(int)
        texts,weights = [],[] 
        for item in items:   
            try:   
                text,weight = [],{}
                if "dseg" in item.keys():
                    for n in item["dseg"]:
                         token = n["key"] 
                         rates[token]+=1                      
                         text.append(token) 
                         weight[token] = n["value"]
                    texts.append(text)
                    weights.append(weight)
            except:
                logger.error(sys.exc_info())
                continue
                        
        #过滤掉只有一次的标签    
        texts= [[token for token in text if rates[token]>1] for text in texts]  
        self.old_textList = texts
        self.old_weightList = weights
        self.dictionary= corpora.Dictionary(texts)   
        
        return self.dictionary
        
    def __getNewCorp(self):
        corpus = [self.dictionary.doc2bow(text) for text in self.old_textList]
        #替换权重      
        k=0
        new_corpus = []        
        for text in corpus:
            tmp=[]
            for n in text:
                c=(n[0],self.old_weightList[k][self.dictionary.get(n[0])])
                tmp.append(c) 
            new_corpus.append(tmp) 
            k+=1         
        self.new_corpus = new_corpus  
        corpus = None
        return new_corpus 
        
    def __calcSim(self,pic=0.02,process=0):
        # time0=time.time()
         self.tfidf_model = models.TfidfModel(self.new_corpus)  
         self.corpus_tfidf = self.tfidf_model[self.new_corpus]      
         sim_t=similarities.Similarity('Similarity-tfidf-index', self.corpus_tfidf, num_features=1000000,num_best=200)
         i=0
         maxAve = 0
         maxIndex = 0 
  #       print(["进程:",process,len(self.source_seg),len(sim_t),self.title,self.trade,self.classes])
         for n in sim_t:
            try:     
                 # ns = [(str(kk[0]),kk[1]) for kk in n if kk[1]>pic]
                 cl = [kk[1] for kk in n if kk[1]>pic] 
                 ave = 0   
                 if len(cl)>0: ave =  stat.mean(cl)      
               # print("sim_t -------:第%s个 平均值：%s 时间:%s " % (str(i),ave,str(time.time()-time0)))             
                 #最短关键字要求
                 if self.source_seg[i] and "dseg" in self.source_seg[i].keys():
                    key_num = len(self.source_seg[i]["dseg"])
                 else:
                    key_num=0 
                 #计算最大相似度及其Index             
                 if key_num > 25 and ave > maxAve :
                     maxAve = ave 
                     maxIndex = i              
                 i+=1
            except:
                 print(["segment erorr:",i,self.source_seg[i]["_id"]])
                 continue       
         self.max_ave = maxAve
         self.max_index = maxIndex     
         return None
    
    def __calcLSi(self):
        pass

from multiprocessing import Pool,Process
class action_similarity:   
    total_amount=0    
    filter = {"count":{"$gt":3}}
    @staticmethod
    def yield_doc(obj,process=-1,p_count=100):
       if process >-1 and p_count>0: 
           docs = obj.col.find(action_similarity.filter,no_cursor_timeout=True).skip(process*p_count).limit(p_count) 
       else:
           docs = obj.col.find(action_similarity.filter,no_cursor_timeout=True) 
           
       print(["进程启动",process,process*p_count,p_count])
       for doc in docs:
            yield doc
           
        
    @staticmethod            
    def mulProcess_similar(process=4):
        t = titleGroup()
        amount = t.col.find(action_similarity.filter).count()
        processPages = int(amount//((process-1)*100))*100
        print(["titleGroup total:",amount,"进程数:",process,"进程amount:",processPages])
        p_list=[]
        for i in range(process):
             p = Process(target=action_similarity.exec_totalSim,args=(i,processPages))
             p_list.append(p)
             p.start()
             
        for j in p_list:
            j.join()   
    
    @staticmethod            
    def mulPool_similar(process=4,poolSize=15):
        t = titleGroup()
        amount = t.col.find(action_similarity.filter).count()
        processPages = int(amount//((process-1)*100))*100
        print(["titleGroup total:",amount,"进程数:",process,"进程amount:",processPages,"进程池大小:",poolSize])      
        pool = Pool(processes=poolSize)
        
        for i in range(process):   
            result = pool.apply_async(action_similarity.exec_totalSim,(i,processPages))
        
        pool.close()
        pool.join()
        if result and result.successful():
            print(['successful',action_similarity.total_amount])               
            
    #整理职位描述       
    @staticmethod 
    def update_des(obj_job,obj_seg,_id):
          doc_job = obj_job.col.find_one({"_id":_id})
          tmp = obj_seg.clean_new(doc_job["des"])
          obj_job.col.update_one({"_id":doc_job["_id"]},{"$set":{"new_des":tmp}})    
           
    @staticmethod
    def exec_totalSim(process=0,processAmount=1000):
        time0 = time.time()
        segs = jobSegment()
        jobs = jobDetail()
        t = titleGroup()
        sub_keys=["trades","classes"]     
        k=0     
        num_error=0
        isMain = True 
        for doc_title in action_similarity.yield_doc(t,process,processAmount):    
            col_seg = segs.col.find({"title": doc_title["title"]})                            
            seg_items=[]
            for seg in col_seg: 
               seg_items.append(seg)                
            print(["文档数量:",doc_title["title"],col_seg.count(),len(seg_items),col_seg.count()==len(seg_items)])
            try:
               #主文档  
             #  if len(seg_items)<10000:
               isMain = True 
               sim = sim_tfidf(seg_items,process)                
               pid = seg_items[sim.max_index]["pid"]
               doc_title["sim_pid"] = pid
               action_similarity.update_des(jobs,segs,pid[1])
                   
               #按行业和职能大类分类计算               
               for key in sub_keys:                
                    keyName="trade"
                    if key == "classes": keyName = "class"
                    for subitem in doc_title[key]:
                        if subitem["count"]>3:
                            sub_docs=[]
                            for doc_seg in seg_items:
                                lists= doc_seg[keyName].split(",")
                                if subitem[keyName] in lists:
                                   sub_docs.append(doc_seg)                                                
                    #        print(["subitem",keyName,subitem[keyName]])        
                            sim = sim_tfidf(sub_docs,process)
                            pid = sub_docs[sim.max_index]["pid"] 
                            subitem["sim_pid"] = pid
                            isMain = False 
                            action_similarity.update_des(jobs,segs,pid[1])
               k+=1               
               t.col.save(doc_title)   
               if k % 200 ==0:  t.log("sim",["sim计算:进程-位置",process,k,"error",num_error,doc_title["title"],len(seg_items),time.time()-time0])
            except:
               logger.error([k,isMain,doc_title["_id"]]) 
        #       logger.error(traceback.print_exc())               
               num_error+=1
               continue 
                
        action_similarity.total_amount+=k    
        t.log("sim",["sim结束 进程-成功-error",process,k,num_error,time.time()-time0])  
            
def main():    
     #action_similarity.exec_totalSim(-1,0)
     #action_similarity.mulProcess_similar(30)
     action_similarity.mulPool_similar(50,30)

if __name__ == "__main__":     
    main()
