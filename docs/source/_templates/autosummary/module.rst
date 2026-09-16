{{ fullname | escape | underline }}

.. currentmodule:: {{ fullname }}

{% if fullname == "homicsx" %}
.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
{% else %}
.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
{% endif %}

{% if functions %}
Functions
---------
{% for item in functions %}
.. autofunction:: {{ fullname }}.{{ item }}
{% endfor %}
{% endif %}

{% if classes %}
Classes
-------
{% for item in classes %}
.. autoclass:: {{ fullname }}.{{ item }}
   :members:
   :undoc-members:
   :show-inheritance:
{% endfor %}
{% endif %}
