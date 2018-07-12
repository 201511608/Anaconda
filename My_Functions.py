#####################  #
# CONTENT
# :: 1  # Common_Functions
# :: 2  # UnSupervised
# :: 3  # Reinforcement
# :: 4  # 
# :: 5  #
# :: 6  #
# :: 7  #
# :: 8  #
# :: 9  #
# :: 10 #
#####################  #


# :: 2 
##################################################################################################################
#####################################################################################
# Common_Functions Class
	# :: 1  Array2DataFrame      #Numpy array to DataFrame
	# :: 2  Array2SaveCSV        #Numpy array to DataFrame to SaveFile



class Common_Functions:
    
	def __init__ (self):
		print("Common_Function Class Instance Begin")

	def __del__(self):
		print("Common_Function Class Instance Destroyed")
	
	def check():  # For No Inputs NoNeed Self
		print("In Common_Function Class")

    #Numpy array to DataFrame
	def Array2DataFrame(self,array):
		import numpy as np
		df = pd.DataFrame(data=array)
		return df

	#Numpy array to DataFrame to SaveFile
	def Array2SaveCSV(self, NpArray,filename='Delete'):
		import numpy as np
		df = pd.DataFrame(data=NpArray)
		df.to_csv(filename)
		


		
# :: 2 
##################################################################################################################
#####################################################################################
# UnSupervised Class
	# :: 1   Scatter_Plot             #plots
	# :: 2   Scatter_Plot_WithCg      #plots
	

class Unsupervised:
    
    def __init__( self):
        print("Unsupervised Instance Begin") 
        
    def __del__(self):
       print("Unsupervised Instance Destroyed")
    
    def check():
        print("In Unsupervised class")
    
	# My UnSupervised Learning Functions
	
    #plots
    def Scatter_Plot(self,x,y,labels=[]):
       # labels=[1 for i in range(len(x))]
        plt.scatter(x,y,c=labels,alpha =0.75)
       # plt.scatter(centroids[:,0],centroids[:,1],color='r',marker='D',s=50)
        plt.show()
        
    def Scatter_Plot_WithCg(self,cgx,cgy,x,y,labels=[]):
        # centroids=model.cluster_centers_  # centroids[:,0],centroids[:,1] 
        #labels=[1 for i in range(len(x))]
        plt.scatter(x,y,c=labels,alpha =0.75)
        plt.scatter(cgx,cgy,color='r',marker='D',s=50)
        plt.show()
        
        
    def Inertia_Cluster_Graph(self,Train_Data):
        sm_data=Train_Data
        ks = range(1, 15)
        inertias = []
        for k in ks:
            # Create a KMeans instance with k clusters: model
            modle=KMeans(n_clusters=k)    
            # Fit model to samples
            modle.fit(sm_data)  
            # Append the inertia to the list of inertias
            inertias.append(modle.inertia_)
        plt.plot(ks, inertias, '-o')
        plt.xlabel('Number of Clusters, k')
        plt.ylabel('Inertia')
        plt.xticks(ks)
        plt.show() 
        
        
    def crosstable(predictedData,targetData):
        import pandas as pd
        ct=pd.crosstab(predictedData,targetData)
        print("\nCrossTable\n",ct)
        
    def Normalizer_Kmean(self,Train_Data,Prediction_Data,clusters=5):    
        ##import numpy as np
        from sklearn.preprocessing import Normalizer
        from sklearn.cluster import KMeans

        # Market_Data=pd.read_csv('sharemarket_data.csv',header=0,usecols= [x for x in range(1,963)])
        ##companiessm_data=np.loadtxt('sharemarket_data.txt') #sharemarket_data
        ##companies=pd.read_csv('sharemarket_data.csv',header=0,usecols= [0]).values.reshape(60)

        #7.1
        normalizer= Normalizer() # Normalizing the Data
        kmeans= KMeans(n_clusters=clusters)   # Change Cluster and check the results
        pipeline = make_pipeline(normalizer,kmeans)
        pipeline.fit(Train_Data)

        #7.2
        prediction =pipeline.predict(Prediction_Data)
        return prediction
    
    def StandardScaler_Kmean(self,Train_Data,Prediction_Data,clusters=5):                     
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            
            # StandardScaler  MaxAbsScalar   Normilizer
            scaler = StandardScaler()
            # Create KMeans instance: kmeans
            kmeans = KMeans(n_clusters=clusters)


            # Create pipeline: pipeline
            pipeline = make_pipeline(scaler,kmeans)  # Make_pipeline
            # Fit the pipeline to samples
            pipeline.fit(samples)
            # Calculate the cluster labels: labels
            labels = pipeline.predict(samples)
            prediction=labels
            return prediction
        
        
    def Dendgrogram_Plot(self,TrainData,labels,method='complete'):
        #label=[1 for i in range(len(TrainData))]
        
        from scipy.cluster.hierarchy import linkage, dendrogram
        import matplotlib.pyplot as plt

        # Calculate the linkage: mergings:: This performs hierarchical clustering
        mergings = linkage(TrainData, method=method)

        # Plot the dendrogram
        dendrogram(mergings,labels=label,leaf_rotation=90,leaf_font_size=6)
        plt.show()




    def DendgrogramDistance_SubPlot(self,TrainData,labels,distance=5,method='complete'):
        from scipy.cluster.hierarchy import linkage
        from scipy.cluster.hierarchy import fcluster

        samples=TrainData
        varieties_=labels
       # varieties=['Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat']

        #Measures distance.
        mergings = linkage(samples,method =method)  # Using above samples

        dendrogram(mergings,labels=varieties_,leaf_rotation=90,leaf_font_size=6,)
        print("Full Dendrogram, No distance limits")
        plt.show()

        #Extracting Intermediate Clusters!
        labels_f = fcluster(mergings,distance,criterion='distance')
        print (labels)
        print ("At distance",distance,"Division considered Above this all assign as Indivudial Number")
        dendrogram(mergings,labels=labels_f,leaf_rotation=90,leaf_font_size=6,)
        plt.show()



    def AllPlots(self,samples,labels,distance):
        
        #Inertia Plot
        Unsupervised.Inertia_Cluster_Graph(self,samples)

        #Scatter Plot                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              +
        Unsupervised.Scatter_Plot(self,samples[:,0],samples[:,1],labels)
        Unsupervised.Scatter_Plot(self,samples[:,2],samples[:,3],labels)

        #DendroGram Plot
        Unsupervised.DendgrogramDistance_SubPlot(self,samples,labels,distance=distance,method='complete')
    
    def Tsne(self,samples,labels,learning_rate=100):
        from sklearn.manifold import TSNE
        model = TSNE(learning_rate=learning_rate)
        tsne_features = model.fit_transform(samples)
        return tsne_features
    def TsnePlot(self,samples,labels_numbers,learning_rate=100):
        print("Sample input Ex :",samples[0,:])
        tsne_features=Unsupervised.Tsne(self,samples,labels,learning_rate=100)
        xs = tsne_features[:,0]
        ys = tsne_features[:,1]
        print("Tsne Output Ex:",tsne_features[0,:])
        plt.scatter(xs, ys, c=labels_numbers)
        plt.show()
    def TsnePlotLabel(self,samples,labels_numbers,labels_name,learning_rate=100):       
        tsne_features=Unsupervised.Tsne(self,samples,labels,learning_rate=100)
        xs = tsne_features[:,0]
        ys = tsne_features[:,1]
        print("Sample input Ex :",samples[0,:])
        print("Tsne Output Ex:",tsne_features[0,:])
        plt.scatter(xs, ys, c=labels_numbers)
        for x, y, company in zip(xs, ys, companies):
            plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
        plt.show()
    
    def Pca_DimensionReduce(self,TrainData,n_components=2):
        #18
        #PCA fit and Transforms  ....  Don't have Predict  . It just Reduce the Dimensions!
        from sklearn.decomposition import PCA

        # Create a PCA model with 2 components: pca
        pca = PCA(n_components=n_components)   # n_components = Components Require !!!

        # Fit the PCA instance to the scaled samples
        pca.fit(TrainData)

        # Transform the scaled samples: pca_features
        pca_features = pca.transform(TrainData)
        
        
        print("Just Reduces the Dimension",TrainData.shape,"->",pca_features.shape)

        return pca_features
    
        
    def PcaPlot(self,TrainData):
        grains=TrainData
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        
        model=PCA()
        model.fit(grains)
        transformed=model.transform(grains)
        ## we can use both to gether  FIT AND TRANSFORM
        #  model.fit_transform(grains)
        #14.1
        print("Actual Data")
        plt.scatter(grains[:,0], grains[:,1])
        #plt.axis('equal')
        plt.show()

        #14.2
        print("\n PCA Transformed \n Data Shift to Mean=0  and Rotates")
        plt.scatter(transformed[:,0], transformed[:,1])
        plt.axis('equal')
        plt.show()

        #print
        print("\n\nActual Data :",grains[0,:])
        print("PCA Transformed Data :",transformed[0,:])
        print("Mean of NotTransformed Data:",model.mean_)
        
        
        mean = model.mean_
        print("\n\nmean :",mean)

        # Get the first principal component: first_pc
        first_pc = model.components_[0,:]
        print("first_pc",first_pc)
        print("The first principal component of the data is the direction in which the data varies the most")
        # Plot first_pc as an arrow, starting at mean
        plt.scatter(grains[:,0], grains[:,1])
        plt.arrow(mean[0],mean[1], first_pc[0], first_pc[1], color='red', width=0.051)
        plt.show()
        
        
    def StandardScalarPca_VariencePlot(self,TrainData):
        #16  PCA PLOT
        # Perform the necessary imports
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        import matplotlib.pyplot as plt

        fish_Data=TrainData

        # Create scaler: scaler
        scaler = StandardScaler()

        # Create a PCA instance: pca
        pca = PCA()

        # Create pipeline: pipeline
        pipeline = make_pipeline(scaler,pca)

        # Fit the pipeline to 'samples'
        pipeline.fit(fish_Data)

        # Plot the explained variances
        features =range(pca.n_components_)    ## HIGHLIGHTING
        print("Features is equal to Input :",features)
        plt.bar(features,pca.explained_variance_)
        plt.xlabel('PCA feature')
        plt.ylabel('variance')
        plt.xticks(features)
        plt.show()
        print(" Plot Shows to what Level Dimension Can be reduce\n Higher the Varience value higher the Results dependence ")
        print (" Select PCA(n_components = Base up on above Graph)")
        
    def Tfidf_ConverttoCsr(self,TrainData):
        #19 
        # return csr_mat
        # tfidf
        # tfidf -> It transforms a list of documents into a word frequency array, which it outputs as a csr_matri
        print("tfidf -> converts to Csr_mat")
        print("csr matrix reduse the space by remembering only non zero entries \n")
        documents= TrainData
        # Import TfidfVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Create a TfidfVectorizer: tfidf
        tfidf = TfidfVectorizer() 

        # Apply fit_transform to document: csr_mat
        csr_mat = tfidf.fit_transform(documents)

        # Print 

        words = tfidf.get_feature_names()
        print()
        print(documents)
        print()
        print("\nLength :",len(words))
        print(words)
        print()
        print("csr_mat.toarray :\n",csr_mat.toarray())
        print()
        print("Csr_Matrix :\n",csr_mat)

        
        # return
        return csr_mat
    
    
    def TruncatedSvd_Kmean(self,TrainData_csr_matrix,titles,Svd_components=50,n_clusters=6):
        #Svd_components -> Dimension Reduction
        #n_clusters -> n_clusters Reducion
        from sklearn.decomposition import TruncatedSVD
        from sklearn.cluster import KMeans
        from sklearn.pipeline import make_pipeline

        # Create a TruncatedSVD instance: svd
        svd = TruncatedSVD(n_components=Svd_components)

        # Create a KMeans instance: kmeans
        kmeans =KMeans(n_clusters=n_clusters)   #

        # Create a pipeline: pipeline
        pipeline = make_pipeline(svd,kmeans)
        
        
        import pandas as pd
        
        articles = TrainData_csr_matrix         # Dirct csr matrix
        titles = titles

       
        import pandas as pd

        # Fit the pipeline to articles
        pipeline.fit(articles)

        # Calculate the cluster labels: labels
        labels = pipeline.predict(articles)
        print(len(labels))
        print("Labels",labels)

        # Create a DataFrame aligning labels and titles: df
        df1 = pd.DataFrame({'label': labels, 'article': titles})


        # Print
        print(df1.sort_values('label'))

    def Nmf(self,TrainData_csr_matrix,titles,n_components=6):
        #21
        # NMF   :: Non Negative Matrix Factorization (NMF)   #for Non Negative
        #  Dimension Reduction technique 
        from sklearn.decomposition import NMF

        #df = pd.read_csv('wikipedia-vectors.csv', index_col=0)
        articles =TrainData_csr_matrix #csr_matrix(df.transpose())         # Dirct csr matrix
        titles =titles  #list(df.columns)

        # Create an NMF instance: model
        model = NMF(n_components=n_components)

        # Fit the model to articles
        model.fit(articles) #articles is CSR_MATRIX

        # Transform the articles: nmf_features
        nmf_features = model.transform(articles)

        # Print the NMF features
        print(nmf_features)

        
        
        # Create a pandas DataFrame: df
        df = pd.DataFrame(nmf_features,index=titles)

        # Print the row for 'Anne Hathaway'
        print(df.iloc[0])

        # Print the row for 'Denzel Washington'
        print(df.iloc[1])
        print("Model.Components_\n",model.components_)
        
    def Nmf_CompImshow(self,TrainData,n_components=7):
       
        samples=TrainData
        # Import NMF
        from sklearn.decomposition import NMF

        # Create an NMF model: model
        model = NMF(n_components=n_components)

        # Apply fit_transform to samples: features
        features = model.fit_transform(samples)

        # Call show_as_image on each component
        for component in model.components_:
            Unsupervised.show_as_image(self,component)

        # Select the 0th row of features: digit_features
        digit_features = features[0,:]

        # Print digit_features
        print(digit_features)

    
    def show_as_image(self,sample):
        from matplotlib import pyplot as plt
        bitmap = sample.reshape((13, 8))
        plt.figure()
        plt.imshow(bitmap, cmap='gray', interpolation='nearest')
        plt.colorbar()
        plt.show()
        
    def Lcd_Imshow(self,samples,x=13,y=8):
        samples = samples #pd.read_csv('lcd-digits.csv',header=None).values  ## IMPORTANT
        # Import pyplot
        from matplotlib import pyplot as plt

        # Select the 0th row: digit
        digit = samples[10,:]

        # Print digit
        print(digit)

        # Reshape digit to a 13x8 array: bitmap
        bitmap = digit.reshape(x,y)

        # Print bitmap
        print(bitmap)

        # Use plt.imshow to display bitmap
        plt.imshow(bitmap, cmap='gray', interpolation='nearest')
        plt.colorbar()
        plt.show()
    
    def NmfNormalizeDot(self,TrainData_Csr,Article_index=0):
        #29
        # give similar article types
        #29.1
        articles=TrainData
        from sklearn.decomposition import NMF

        # Create an NMF instance: model
        model = NMF(n_components=6)

        # Fit the model to articles
        model.fit(articles)

        # Transform the articles: nmf_features
        nmf_features = model.transform(articles)

        #
        # Perform the necessary imports
        import pandas as pd
        from sklearn.preprocessing import normalize

        # Normalize the NMF features: norm_features
        norm_features =normalize(nmf_features)
        # Create a DataFrame: df
        df =pd.DataFrame(norm_features,titles)

        # Select the row corresponding to 'Cristiano Ronaldo': article
        article = df.iloc[Article_index]     # TO  GET SIMILAR TYPE OF ARTICLES !!!

        # Compute the dot products: similarities
        similarities = df.dot(article)

        # Display those with the largest cosine similarity
        print(similarities.nlargest())
 

 
 		
# :: 3
##################################################################################################################
#####################################################################################
# Reinforcement Class
	# :: 1  Reinforcement_PoleBalance  
	# :: 2   
 

class Reinforcement:
    def __init__( self):
        print("Reinforcement Class Instance Begin ") 

    def __del__(self):
	    print("Reinforcement Class Instance Destroyed")

    def check():
	    print("In Reinforcement class")
		
	### PoleBalance	
    class Reinforcement_PoleBalance:
        # Defautl Goal_steps , LR, score_requirement and initia_games Defined  
        def __init__( self,goal_steps = 500,LR = 1e-3,score_requirement = 100,initial_games = 10000):
            print("Reinforcement_PoleBalance Class Instance Begin :: Takes default inputs") 
            self.LR = LR
            self.goal_steps = goal_steps
            self.score_requirement = score_requirement
            self.initial_games = initial_games 


        def __del__(self):
            print("Reinforcement_PoleBalance Class Instance Destroyed")

        def check():
            print("In Reinforcement_PoleBalance class")

         # 1.1
        # Pole Balance
        # Reinforcement 
        # Make it Better
        def Pole_Balance(self):
            import gym
            import random
            import numpy as np
            import tflearn
            from tflearn.layers.core import input_data, dropout, fully_connected
            from tflearn.layers.estimator import regression
            from statistics import median, mean
            from collections import Counter

            LR = 1e-3
            env = gym.make("CartPole-v0")
            env.reset()
            goal_steps = 500
            score_requirement = 100
            initial_games = 10000
            return score_requirement

        def some_random_games_first(self):
            # Each of these is its own game.
            for episode in range(5):
                env.reset()
                # this is each frame, up to 200...but we wont make it that far.
                for t in range(200):
                    # This will display the environment
                    # Only display if you really want to see it.
                    # Takes much longer to display it.
                    env.render()

                    # This will just create a sample action in any environment.

    # In this environment, the action can be 0 or 1, which is left or right
                    action = env.action_space.sample()

                    # this executes the environment with an action, 
                    # and returns the observation of the environment, 
                    # the reward, if the env is over, and other info.
                    observation, reward, done, info = env.step(action)
                    if done:
                        break

        #some_random_games_first()
        def initial_population(self):
            import numpy as np
            import random
            import gym
            from statistics import median, mean
            from collections import Counter

            env = gym.make("CartPole-v0")
            env.reset()

            # [OBS, MOVES]
            training_data = []
            # all scores:
            scores = []
            # just the scores that met our threshold:
            accepted_scores = []
            # iterate through however many games we want:
            for _ in range(self.initial_games):
                score = 0
                # moves specifically from this environment:
                game_memory = []
                # previous observation that we saw
                prev_observation = []
                # for each frame in 200
                for _ in range(self.goal_steps):
                    # choose random action (0 or 1)
                    action = random.randrange(0,2)
                    # do it!
                    observation, reward, done, info = env.step(action)

                    # notice that the observation is returned FROM the action
                    # so we'll store the previous observation here, pairing
                    # the prev observation to the action we'll take.
                    if len(prev_observation) > 0 :
                        game_memory.append([prev_observation, action])
                    prev_observation = observation
                    score+=reward
                    if done: break

                # IF our score is higher than our threshold, we'd like to save
                # every move we made
                # NOTE the reinforcement methodology here. 
                # all we're doing is reinforcing the score, we're not trying 
                # to influence the machine in any way as to HOW that score is 
                # reached.
                if score >= self.score_requirement:
                    accepted_scores.append(score)
                    for data in game_memory:
                        # convert to one-hot (this is the output layer for our neural network)
                        if data[1] == 1:
                            output = [0,1]
                        elif data[1] == 0:
                            output = [1,0]

                        # saving our training data
                        training_data.append([data[0], output])

                # reset env to play again
                env.reset()
                # save overall scores
                scores.append(score)

            # just in case you wanted to reference later
            training_data_save = np.array(training_data)
            np.save('saved1.npy',training_data_save)

            # some stats here, to further illustrate the neural network magic!
            print('Average accepted score:',mean(accepted_scores))
            print('Median score for accepted scores:',median(accepted_scores))
            print(Counter(accepted_scores))

            return training_data



        def neural_network_model(self, input_size):
            import tflearn
            from tflearn.layers.core import input_data, dropout, fully_connected
            from tflearn.layers.estimator import regression

            network = input_data(shape=[None, input_size, 1], name='input')

            network = fully_connected(network, 128, activation='relu')
            network = dropout(network, 0.8)

            network = fully_connected(network, 256, activation='relu')
            network = dropout(network, 0.8)

            network = fully_connected(network, 512, activation='relu')
            network = dropout(network, 0.8)

            network = fully_connected(network, 512, activation='relu')
            network = dropout(network, 0.8)

            network = fully_connected(network, 512, activation='relu')
            network = dropout(network, 0.8)

            network = fully_connected(network, 2, activation='softmax')
            network = regression(network, optimizer='adam', learning_rate=self.LR, loss='categorical_crossentropy', name='targets')
            model = tflearn.DNN(network, tensorboard_dir='log')

            return model
        
        def neural_network_model_initialize (self,training_data):
            import numpy as np
            X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
            y = [i[1] for i in training_data]
            model = Reinforcement.Reinforcement_PoleBalance.neural_network_model(self,input_size = len(X[0]))
            return model

        def train_model(self,training_data, model):
            import numpy as np
        
            X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
            y = [i[1] for i in training_data]
            model.fit({'input': X}, {'targets': y}, n_epoch=3, snapshot_step=500, show_metric=True, run_id='openai_learning')
            return model

        # model = train_model(initial_population())

        def Pole_test(self,model):
            # 1.2
            import gym
            import random
            import numpy as np
            import tflearn
            from tflearn.layers.core import input_data, dropout, fully_connected
            from tflearn.layers.estimator import regression
            from statistics import median, mean
            from collections import Counter

            LR = 1e-3
            env = gym.make("CartPole-v0")
            env.reset()
            goal_steps = 500
            score_requirement = 50
            initial_games = 10000
        #    def     training_data =initial_population()
        #         model = train_model(training_data)
            scores = []
            choices = []

            for each_game in range(10):
                score = 0
                game_memory = []
                prev_obs = []
                env.reset()
                for _ in range(goal_steps):
                    env.render()
                    if len(prev_obs)==0:
                        action = random.randrange(0,2)
                    else:
                        action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
                    choices.append(action)               
                    new_observation, reward, done, info = env.step(action)
                    prev_obs = new_observation
                    game_memory.append([new_observation, action])
                    score+=reward
                    if done: break
                scores.append(score)
            env.close()
            print('Average Score:',sum(scores)/len(scores))
            print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
            print(score_requirement)
        
#       #  Function Call ::  Reinforcement_PoleBalance
#       #  r=Reinforcement.Reinforcement_PoleBalance()
#       #  training_data=r.initial_population()
#       #  model = r.neural_network_model_initialize (training_data)
#       #  model=r.train_model(training_data,model)
#       #  r.Pole_test(model)                      