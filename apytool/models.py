from django.db import models

# Create your models here.

class ShrimpData(models.Model):
	species_choices = (
	  ('Tiger Shrimp', 'Tiger Shrimp'),
      ('L. Vannamei' , 'L. Vannamei')
	 )

	state_choices   = (
	 ('West Bengal'   , 'West Bengal'),
	 ('Orissa'        , 'Orissa'),
	 ('Andhra Pradesh', 'Andhra Pradesh'),
	 ('Tamil Nadu'    , 'Tamil Nadu'),
	 ('Kerela'        , 'Kerela'),
	 ('Karnataka'     , 'Karnataka'),
	 ('Goa'           , 'Goa'),
	 ('Maharastra'    , 'Maharastra'),
	 ('Gujarat'       , 'Gujarat')
	)
	species    = models.CharField(max_length=30, choices=species_choices, blank=False)
	State      = models.CharField(max_length=30, choices=state_choices, blank=False)
	Year       = models.CharField(max_length=10, blank=False)
	Area       = models.PositiveIntegerField(blank=False)
	Production = models.PositiveIntegerField(blank=False)


	def __str__(self):
		return self.species+self.State+self.Year


class Discrepancie(models.Model):
	"""docstring for Discrepancie"""
	Email      = models.EmailField(max_length=254, blank=False)
	Issues     = models.TextField(blank=False)
	def __str__(self):
		return self.Email+self.id