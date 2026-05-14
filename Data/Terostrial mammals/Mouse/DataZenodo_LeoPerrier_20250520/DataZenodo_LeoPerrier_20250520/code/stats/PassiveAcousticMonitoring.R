library(ggplot2)
library(glue)
library(dplyr)
library(forcats)

pam = read.table('D:/pam/all_datasets/pam.csv', header = T, sep = ';', dec = ',')
bushes_codes = readxl::read_excel('D:/pam/bushes_codes.xlsx')

for (i in 1:nrow(pam)) {
  pam$group[i] = bushes_codes$group[bushes_codes$exact_bush == pam$Bush[i]]
  pam$x[i] = bushes_codes$x[bushes_codes$exact_bush == pam$Bush[i]]
  pam$y[i] = bushes_codes$y[bushes_codes$exact_bush == pam$Bush[i]]
  pam$place[i] = bushes_codes$place[bushes_codes$exact_bush == pam$Bush[i]]
  if (i%%10000 == 0) {print(i)}
}
# 
# # adding time of the day in hours
# pam$hour = as.integer(pam$Exact.Time..s./3600)
# 
# #adding que day
# pam$day = substr(pam$Image_ID, 1, 8)


pam3 = pam
pam3$place[pam3$place=='evening'|pam3$place=='morning'] = 'inside'

table(pam[pam$place=='inside',]$Bush, pam[pam$place=='inside',]$day)
ag_place = aggregate(ID~call_type+place, data = pam3, FUN=length)
table(pam3$place)
mean(ag_place$ID[ag_place$place=='inside'])
sd(ag_place$ID[ag_place$place=='inside'])

# create percentages df with zeros
ag_place = aggregate(ID~call_type+place+day, data = pam3, FUN=length)
colnames(ag_place)[length(colnames(ag_place))]='n'

calls_place_combinations <- expand.grid(call_type = unique(ag_place$call_type), 
                                        place = unique(ag_place$place), 
                                        day = unique(ag_place$day))
perc_place = transform(ag_place, perc = ave(n, place, day, FUN = prop.table))
perc_place = merge(perc_place, calls_place_combinations, all.y = T)
perc_place$perc[is.na(perc_place$perc)] = 0



# plot
perc_place$call_type = fct_relevel(perc_place$call_type, 'longdown','down', 'modulated', 'up', "flat")
perc_place$place = fct_relevel(perc_place$place, 'inside', 'intra', "inter")
perc_place$nicelabs = NA
perc_place$shortlabs = NA

for (i in 1:nrow(perc_place)) {
  if (perc_place$place[i] == 'inside') {perc_place$nicelabs[i]="Nest\nbush"; perc_place$shortlabs[i]="NB"}
  if (perc_place$place[i] == 'intra') {perc_place$nicelabs[i]="Within\nterritory"; perc_place$shortlabs[i]="WT"}
  if (perc_place$place[i] == 'inter') {perc_place$nicelabs[i]="Between\nterritories"; perc_place$shortlabs[i]="BT"}
  if (i%%10000 == 0) {print(i)}
}

perc_place$nicelabs = fct_relevel(perc_place$nicelabs, 
                                  'Nest\nbush', 'Within\nterritory', 'Between\nterritories')

perc_place$shortlabs = fct_relevel(perc_place$shortlabs, 
                                   'NB', 'WT', 'BT')



ggplot(data = perc_place, aes(x = perc/15, y = nicelabs, fill = call_type))+#, color = call_type)) + 
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("flat" = "#ff595e",
                               "up"="#ff924c",
                               "modulated"="#FFCA3A",
                               "down" = "#A9B858",
                               "longdown"="#1982c4")) +
  # scale_color_manual(values = c("flat" = "#ff595e",
  #                               "up"="#ff924c",
  #                               "modulated"="#FFCA3A",
  #                               "down" = "#A9B858",
  #                               "longdown"="#1982c4")) +
  labs(title = "Call types distribution per recording site", y = "Recording site", x = "Proportion of calls") +
  guides(fill=guide_legend(title="Vocalization type"), color = "none")+
  theme_classic() +
  coord_flip() 

# ggsave(filename = "D:/pam/all_plots/voc_type_repartition_per_recording_site2.png", width = 5, height = 3.5)


ggplot(data = perc_place, aes(x = perc, y = shortlabs, fill = call_type, color = call_type)) + 
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("flat" = "#ff595e",
                               "up"="#ff924c",
                               "modulated"="#FFCA3A",
                               "down" = "#A9B858",
                               "longdown"="#1982c4")) +
  scale_color_manual(values = c("flat" = "#ff595e",
                                "up"="#ff924c",
                                "modulated"="#FFCA3A",
                                "down" = "#A9B858",
                                "longdown"="#1982c4")) +
  labs(title = "Call types repartition per day", y = "Recording site", x = "Percentage of calls") +
  guides(fill=guide_legend(title="Vocalization type"), color = "none" )+
  theme_classic() +
  coord_flip() +
  facet_wrap(~day, nrow = 5, ncol = 3)

#ggsave(filename = "D:/pam/all_plots/voc_type_repartition_per_recording_site_per_day_no-SR-SS.png", width = 10, height = 10)

# modèles

library(rstan)
library(brms)


hist(perc_place$perc)

# format to use cbind() model 

an_perc_place = expand.grid(shortlabs = unique(perc_place$shortlabs), 
                            day = unique(perc_place$day),
                            flat = 0, up = 0, down = 0, modulated = 0, longdown = 0)

for (i in 1:nrow(an_perc_place)) {
  an_perc_place$flat[i] = perc_place$perc[perc_place$shortlabs == an_perc_place$shortlabs[i] &
                                            perc_place$day == an_perc_place$day[i] &
                                            perc_place$call_type == 'flat']
  an_perc_place$up[i] = perc_place$perc[perc_place$shortlabs == an_perc_place$shortlabs[i] &
                                          perc_place$day == an_perc_place$day[i] &
                                          perc_place$call_type == 'up']
  an_perc_place$down[i] = perc_place$perc[perc_place$shortlabs == an_perc_place$shortlabs[i] &
                                            perc_place$day == an_perc_place$day[i] &
                                            perc_place$call_type == 'down']
  an_perc_place$modulated[i] = perc_place$perc[perc_place$shortlabs == an_perc_place$shortlabs[i] &
                                                 perc_place$day == an_perc_place$day[i] &
                                                 perc_place$call_type == 'modulated']
  an_perc_place$longdown[i] = perc_place$perc[perc_place$shortlabs == an_perc_place$shortlabs[i] &
                                                perc_place$day == an_perc_place$day[i] &
                                                perc_place$call_type == 'longdown']
}

# adding 2e-16 constant to use dirichlet distribution
an_perc_place[,3:7] = an_perc_place[,3:7] + 2e-16

# -------------------------- MODEL BUILDING CODE --------------------------
#
mod_perc_pam = brms::brm(mvbind(flat, up, modulated, down, longdown) ~ shortlabs + (1|day), 
                         data = an_perc_place, family = 'dirichlet', 
                         warmup = 500, iter = 4000, 
                         chains = 4, cores = 4,
                         file = 'D:/pam/all_models/mod_all_perc_pam_dirichlet_no-SR-SS.rds')
# 
summary(mod_perc_pam)
# 
plot(mod_perc_pam)
# 
conditional_effects(mod_perc_pam, categorical = T)
# 
#
#  ------------------------------------------------------------------------

# plots of fitted values
newdata = data.frame(shortlabs = c('IN', 'InT', 'TB'))
fit = fitted(mod_perc_pam, newdata = newdata, robust = T, re_formula = NA, summary = T)

j = 1
for (i in as.character(levels(as.factor(perc_place$call_type)))) {
  # voc = gsub("_", "", i)
  voc = paste0('P(Y = ', i, ')')
  pl_fit_voc = as.data.frame(fit[, , voc])
  pl_fit_voc$voc_type = i
  pl_fit_voc = cbind(newdata, pl_fit_voc)
  if (j==1) {pl_fit = pl_fit_voc}
  else {pl_fit = rbind(pl_fit, pl_fit_voc)}
  j=j+1
}

colnames(pl_fit) = c('place', 'fit', 'se', 'lwr', 'upr', 'voc_type')
pl_fit$place = c('NB', 'WT', 'BT')
pl_fit$place = fct_relevel(pl_fit$place, 'NB', 'WT', 'BT')
pl_fit$leg_place[pl_fit$place=='NB'] = 'NB - Nest bush'
pl_fit$leg_place[pl_fit$place=='WT'] = 'WT - Within territory'
pl_fit$leg_place[pl_fit$place=='BT'] = 'BT - Between territories'

for (voc in unique(pl_fit$voc_type)) {
  ggplot(data = pl_fit[pl_fit$voc_type==voc,]) +
    geom_pointrange(aes(x = place, y = fit, ymin = lwr, ymax = upr, color = leg_place, shape = leg_place), size = 0.5)  +
    # scale_color_discrete(labels = c("Inside\nnest", "Intra-\nterritory", "Territory\nborder"))+
    # scale_shape_discrete(labels = c("Inside\nnest", "Intra-\nterritory", "Territory\nborder"))+
    scale_color_manual(values = c("NB - Nest bush" = "#ed7d31",
                                  "WT - Within territory"="#ffbf00",
                                  'BT - Between territories'="#c00000")) +
    scale_shape_manual(values = c("NB - Nest bush" = 17, "WT - Within territory"= 15, "BT - Between territories"= 16)) +
    scale_y_continuous(labels = scales::label_number(accuracy = 0.001)) +
    theme_bw()+
    theme(panel.grid = element_blank()) +
    labs(y = '% of vocalizations', x = 'Recording site', title = voc)+
    #theme(legend.position = 'none') 
    guides(color=guide_legend(title="Recording site"), shape=guide_legend(title="Recording site"))
    
  plot(last_plot()) 
  
 ggsave(paste0('D:/pam/all_plots/voc_types_per_site_no-SR-SS/LEGEND_TEMPLATE2_', voc, '.png'), width=6, height = 2)
}

#write.csv2(pam, file = 'D:/pam/all_datasets/all_calls_with_group.csv', row.names = F)

# contrasts 

newdata = data.frame(shortlabs = c('IN', 'InT', 'TB'))
fit = fitted(mod_perc_pam, newdata = newdata, robust = T, re_formula = NA, summary = F)
d = fit[, 2, 'flat'] - fit[, 3, 'flat']
qs = round(quantile(d, c(0.025,0.5,0.975))*100, 2)
glue('{qs[2]}%, [{qs[1]}%; {qs[3]}%]')
