from .models import ShrimpData, Discrepancie
from django.forms import ModelForm

#declaring form through the model 
class DetailForm(ModelForm):
	class Meta:
		model  = ShrimpData
		fields = ('species', 'State')

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.fields['species'].widget.attrs.update({'class': 'form-control'})
		self.fields['State'].widget.attrs.update({'class': 'form-control'})
		

class discrepancyForm(ModelForm):
	class Meta(object):
		"""docstring for Meta"""
		model    = Discrepancie
		fields   = ('Email', 'Issues')

			