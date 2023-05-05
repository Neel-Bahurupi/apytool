# Generated by Django 4.0.1 on 2022-01-13 17:50

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ShrimpData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('species', models.CharField(choices=[('Tiger Shrimp', 'Tiger Shrimp'), ('L. Vannamei', 'L. Vannamei'), ('Scampi', 'Scampi')], max_length=30)),
                ('State', models.CharField(choices=[('West Bengal', 'West Bengal'), ('Orissa', 'Orissa'), ('Andhra Pradesh', 'Andhra Pradesh'), ('Tamil Nadu', 'Tamil Nadu'), ('Kerela', 'Kerela'), ('Karnataka', 'Karnataka'), ('Goa', 'Goa'), ('Maharastra', 'Maharastra'), ('Gujarat', 'Gujarat')], max_length=30)),
                ('Year', models.CharField(max_length=10)),
                ('Area', models.PositiveIntegerField()),
                ('Production', models.PositiveIntegerField()),
            ],
        ),
    ]