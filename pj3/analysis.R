library(tidyverse)
library(ggplot2)
library(ggpubr)
library(hrbrthemes)
library(viridis)
library(ggthemes)


data1 <- read_csv("C://Users//guozh//Desktop//CS-375//PJ3//new_results.csv", col_names = FALSE)

# w/ Max tree size = 75, Tournament Size = 7
colnames(data1) <- c("Max_Gen", "Pop_Size", "Variation_Rate", "Test_Set", "Max_Depth", "Total_Error")

data1 <- transform(data1, Variation_Rate = ifelse(Variation_Rate == "[0.7, 0.2, 0.1]", "7/2/1", "2/7/1"))

data1 <- transform(data1, Test_Set = ifelse(Test_Set == "project_test_func.tsv", "Naive", "Vineyard"))

col_target <- c("Max_Gen", "Pop_Size", "Variation_Rate", "Test_Set", "Max_Depth")
data1[col_target] <- lapply(data1[col_target], as.factor)

# G1: Total Completion, based on Test sets
overall <- ggplot(data = data1) +
    geom_histogram(aes(x = Test_Set, fill = Test_Set), stat="count", width  = 0.5, show.legend = FALSE) +
    scale_y_continuous(breaks = round(seq(min(0), max(25), by = 2))) +
    labs(title="Total Number of Completion, 24 Runs for Each Testing Datas Set") + 
    xlab("Data Set Used") +
    ylab("Successfully Finished") +
    theme_linedraw() +
    theme(plot.title = element_text(hjust = 0.5))

error_mean <- ggplot(data = data1, aes(x = Test_Set, y = Total_Error)) +
        geom_boxplot(width = .6, outlier.shape = NA) +
        geom_jitter(shape=16, position=position_jitter(0.3)) +
        stat_summary(fun=mean, geom="point", shape=24, size=2, fill = "Red") + 
        theme_light() + 
        labs(title="Total Errors for Best Programs Found, Mean Shown as Red") + 
        xlab("Data Set Used") +
        ylab("Best Total Errors at the End") +
        theme(plot.title = element_text(hjust = 0.5))
    


ggarrange(overall, error_mean, ncol = 2, nrow = 1)



# G2: Model
md <- lm(Total_Error ~ Max_Gen + Pop_Size + Variation_Rate + Test_Set + Max_Depth, data = data1)
summary(md)


# G3: Population Size
pop_size <- ggplot(data = data1, aes(x = Pop_Size, y = Total_Error, fill = Pop_Size)) +
        geom_violin(width = .6, color = "black", size = 1.3) +
        geom_boxplot(width=0.1, color = "black") +
        scale_fill_viridis(discrete = TRUE, alpha=0.5, option="A") +
        stat_summary(fun=mean, geom="point", shape=23, size=4, fill = "Red") +
        scale_y_continuous(limits = c(0, 200)) +
        theme_ipsum() +
        theme_light() +
        labs(title="Total Errors for Best Programs Found (Mean Shown in Red)") + 
        xlab("Population Size") +
        ylab("Ending Total Errors") +
        theme(plot.title = element_text(hjust = 0.5))

# G4 Generation & population
data2 <- aggregate(x=data1$Total_Error,
                   by=list(data1$Max_Gen,data1$Pop_Size),
                   FUN=mean)
colnames(data2) <- c( "Max_Gen", "Pop_Size", "Mean_TE")

gen_pop <- ggplot(data2, aes(x = Pop_Size, y = Mean_TE, fill = Max_Gen)) +   
    geom_bar(position = "dodge", stat="identity") +
    geom_text(aes(label=round(Mean_TE)), position=position_dodge(width=0.9), vjust=-0.25) +
    labs(title="Results for Different Population Size and Generation") + 
    xlab("Population Size") +
    ylab("Ending Total Errors") +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme_economist()

# G5: Variation, Max Depth

data3 <- aggregate(x=data1$Total_Error,
                   by=list(data1$Variation_Rate,data1$Max_Depth),
                   FUN=mean)
colnames(data3) <- c( "Variation_Rate", "Max_Depth", "Mean_TE")

var_dep <- ggplot(data3, aes(x = Variation_Rate, y = Mean_TE, fill = Max_Depth)) +   
    geom_bar(position = "dodge", stat="identity") +
    geom_text(aes(label=round(Mean_TE)), position=position_dodge(width=0.9), vjust=1.5, color = "white") +
    labs(title="Results for Different Maximum Depth and Variation Ratio") + 
    xlab("Variations: Crossover/Mutation/Reproduction") +
    ylab("Ending Total Errors") +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme_stata() + scale_fill_stata()

# G6: Depth Distribution

data4 <- aggregate(x=data1$Total_Error,
                   by=list(data1$Max_Depth),
                   FUN=mean)

dep_te <- ggplot(data = data1,aes(x = Total_Error, color=Max_Depth, fill=Max_Depth)) +
    geom_histogram(alpha=0.3, position="identity", binwidth = 10, size = 1.2) +
    geom_vline(data=data4, aes(xintercept=x, color=Group.1),
                   linetype=3, alpha = 1, size = 1.4) +
        scale_color_brewer(palette="Dark2")+
        scale_fill_brewer(palette="Dark2") +
        labs(title="Overall Distribution of Total Error, Grouped by Maximum Depth") + 
        xlab("Total Errors") +
        ylab("Count") +
        theme(plot.title = element_text(hjust = 0.5)) +
        theme(legend.position="top") +
        theme_classic()
    

