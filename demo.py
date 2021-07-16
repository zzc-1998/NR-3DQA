from feature_extract import get_feature_vector
#demo
objpath = "hhi_5.ply"
features = get_feature_vector(objpath)

#show the features
cnt = 0
for feature_domain in ['l','a','b','curvature','anisotropy','linearity','planarity','sphericity']:
    for param in ["mean","std","entropy","ggd1","ggd2","aggd1","aggd2","aggd3","aggd4","gamma1","gamma2"]:
        print(feature_domain + "_" + param + ": " + str(features[cnt]))
        cnt = cnt + 1
